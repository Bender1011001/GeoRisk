"""
Physics-Informed Neural Network for Sinkhole Detection
========================================================

Adapted from GeoAnomalyMapper's gravity inversion system for detecting
mass deficits (voids/cavities) associated with sinkhole formation.

This module inverts gravity anomaly data to produce density contrast maps,
specifically tuned for identifying negative density anomalies (voids).

Scientific Basis:
- Parker's Oldenburg formula in frequency domain
- Void mode biases toward negative mass detection
- Threshold calibration based on 2020-2025 literature
"""

import os
import time
import logging
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# Configuration Constants for Sinkhole Detection
# ==========================================
@dataclass
class SinkholeInversionConfig:
    """Configuration for sinkhole-focused gravity inversion."""
    GPU_ID: int = 0
    SEED: int = 42
    LR: float = 1e-3
    EPOCHS: int = 1000
    BATCH_SIZE: int = 1
    TILE_SIZE: int = 1024
    
    # Depth parameters - tuned for shallow sinkhole cavities
    DEPTH_ESTIMATE: float = 50.0      # Mean depth of cavities (meters) - shallower than mineral targets
    THICKNESS_ESTIMATE: float = 30.0  # Thickness of the cavity zone (meters)
    
    # Density parameters
    MAX_DENSITY: float = 500.0        # Max density contrast (kg/m^3)
    VOID_DENSITY: float = -2670.0     # Typical void deficit vs limestone (kg/m³)
    
    # Loss weights - optimized for void detection
    PHYSICS_WEIGHT: float = 10.0
    SPARSITY_WEIGHT: float = 0.001    
    TV_WEIGHT: float = 0.01          
    VOID_BIAS_WEIGHT: float = 0.5     # Penalty for positive mass (we want voids)
    
    USE_AMP: bool = True
    
    # Region-specific thresholds (based on literature)
    REGION_THRESHOLDS: Dict = None
    
    def __post_init__(self):
        self.REGION_THRESHOLDS = {
            'florida_karst': {
                'velocity_yellow': -3.0,   # mm/yr
                'velocity_red': -6.0,      # mm/yr
                'sensor': 'X-band',
                'description': 'Cover-collapse sinkholes in sand/limestone'
            },
            'texas_salt': {
                'velocity_yellow': -10.0,
                'velocity_red': -20.0,
                'sensor': 'C-band',
                'description': 'Salt dome dissolution features'
            },
            'dead_sea': {
                'velocity_yellow': -15.0,
                'velocity_red': -20.0,
                'sensor': 'C-band',
                'description': 'Evaporite dissolution with viscoelastic overburden'
            },
            'konya_basin': {
                'velocity_yellow': -15.0,
                'velocity_red': -30.0,
                'sensor': 'C-band',
                'description': 'Anthropogenic groundwater depletion sinkholes'
            },
            'general': {
                'velocity_yellow': -5.0,
                'velocity_red': -10.0,
                'sensor': 'C-band',
                'description': 'Generic karst terrain'
            }
        }


DEFAULT_CONFIG = SinkholeInversionConfig()


def get_device(config: SinkholeInversionConfig) -> Tuple[torch.device, bool]:
    """Robustly determine the computation device."""
    if os.environ.get("FORCE_CPU_INVERSION", "0") == "1":
        logger.info("Forcing CPU execution based on environment variable.")
        return torch.device("cpu"), False
    
    if torch.cuda.is_available():
        device_id = config.GPU_ID
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        return device, config.USE_AMP
    else:
        logger.warning("CUDA not available. Falling back to CPU.")
        return torch.device("cpu"), False


# ==========================================
# Physics Layer - Void Detection Mode
# ==========================================
class VoidGravityPhysicsLayer(nn.Module):
    """
    Differentiable Forward Gravity Modeling for Void Detection.
    
    Implements Parker's Oldenburg formula optimized for detecting
    shallow mass deficits (cavities, voids) typical of sinkhole formation.
    
    Key modifications from mineral detection:
    - Shallower depth estimates (10-100m vs 200-1000m)
    - Negative density contrast focus
    - Smaller spatial wavelength sensitivity
    """
    
    def __init__(self, pixel_size_meters: float, mean_depth: float = 50.0, thickness: float = 30.0):
        super().__init__()
        self.pixel_size = pixel_size_meters
        self.depth = mean_depth
        self.thickness = thickness
        self.G = 6.674e-11
        self.SI_to_mGal = 1e5
        self.earth_filter = None
        
        logger.info(f"VoidGravityPhysics: depth={mean_depth}m, thickness={thickness}m, pixel={pixel_size_meters:.1f}m")
    
    def forward(self, density_map: torch.Tensor) -> torch.Tensor:
        """
        Forward gravity modeling from density to gravity anomaly.
        
        Args:
            density_map: [B, C, H, W] Density contrast in kg/m³ (negative = void)
            
        Returns:
            [B, C, H, W] Simulated gravity anomaly in mGal
        """
        B, C, H, W = density_map.shape
        
        # FFT-friendly padding (power of 2)
        target_h = 1 << (H + int(H * 0.25) - 1).bit_length()
        target_w = 1 << (W + int(W * 0.25) - 1).bit_length()
        
        if target_h < H * 1.25: target_h *= 2
        if target_w < W * 1.25: target_w *= 2
        
        pad_total_h = target_h - H
        pad_total_w = target_w - W
        
        pad_h_top = pad_total_h // 2
        pad_h_bottom = pad_total_h - pad_h_top
        pad_w_left = pad_total_w // 2
        pad_w_right = pad_total_w - pad_w_left
        
        density_padded = F.pad(density_map, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='reflect')
        B, C, H_pad, W_pad = density_padded.shape
        
        # Precompute filter (lazy loading)
        if (self.earth_filter is None) or (self.earth_filter.shape[-2:] != (H_pad, W_pad)):
            device = density_map.device
            
            freq_y = torch.fft.fftfreq(H_pad, d=self.pixel_size).to(device)
            freq_x = torch.fft.fftfreq(W_pad, d=self.pixel_size).to(device)
            
            KY, KX = torch.meshgrid(freq_y, freq_x, indexing='ij')
            K_magnitude = torch.sqrt(KX**2 + KY**2)
            k_angular = 2 * np.pi * K_magnitude
            
            # Parker's Formula for shallow void detection
            filter_response = (2 * np.pi * self.G * torch.exp(-k_angular * self.depth) * self.thickness)
            self.earth_filter = filter_response.unsqueeze(0).unsqueeze(0)
        
        # FFT Convolution
        d_fft = torch.fft.fft2(density_padded)
        g_fft = d_fft * self.earth_filter
        gravity_pred_padded = torch.real(torch.fft.ifft2(g_fft))
        
        # Crop to original size
        gravity_pred = gravity_pred_padded[:, :, pad_h_top:pad_h_top+H, pad_w_left:pad_w_left+W]
        
        return gravity_pred * self.SI_to_mGal


# ==========================================
# Neural Network Architecture
# ==========================================
class DoubleConv(nn.Module):
    """(Conv -> InstanceNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SinkholePINN(nn.Module):
    """
    U-Net architecture optimized for sinkhole (void) detection.
    
    Key features:
    - Biased toward negative density outputs (voids)
    - Calibrated for shallow cavity detection
    - Outputs bounded density contrast map
    """
    
    def __init__(self, max_density: float = 500.0):
        super().__init__()
        self.max_density = max_density
        
        # Encoder
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(128 + 64, 64)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(64 + 32, 32)
        
        self.outc = nn.Conv2d(32, 1, kernel_size=1)
        
        # Initialize with slight negative bias for void detection
        nn.init.constant_(self.outc.bias, -0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        if x.shape != x3.shape: x = F.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x2.shape: x = F.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x1.shape: x = F.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        # Tanh bounded output with full range for both voids (-) and infill (+)
        return torch.tanh(self.outc(x)) * self.max_density


# ==========================================
# Main Inversion Function
# ==========================================
def invert_gravity_for_voids(
    tif_path: str,
    output_path: str,
    config: Optional[SinkholeInversionConfig] = None,
    region: str = 'florida_karst'
) -> Dict:
    """
    Perform gravity inversion specifically tuned for void/cavity detection.
    
    Args:
        tif_path: Path to input Bouguer gravity anomaly GeoTIFF
        output_path: Path for output density contrast GeoTIFF
        config: Inversion configuration (uses defaults if None)
        region: Geological region for threshold calibration
        
    Returns:
        Dictionary containing inversion metadata and statistics
    """
    if config is None:
        config = SinkholeInversionConfig()
    
    logger.info(f"Starting VOID-MODE inversion for region: {region}")
    region_config = config.REGION_THRESHOLDS.get(region, config.REGION_THRESHOLDS['general'])
    logger.info(f"Region config: {region_config}")
    
    # Setup Device
    device, use_amp = get_device(config)
    
    # Load Data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        profile = src.profile
        height, width = data.shape
        
        # Calculate pixel size
        if src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            deg_to_meter = 111320 * np.cos(np.deg2rad(center_lat))
            pixel_size = src.transform[0] * deg_to_meter
        else:
            pixel_size = src.transform[0]
            
        logger.info(f"Loaded {tif_path}: {data.shape}, Pixel Size: {pixel_size:.1f}m")
    
    # Preprocessing - Z-score normalization
    grav_mean = np.nanmean(data)
    grav_std = np.nanstd(data)
    if grav_std == 0: grav_std = 1
    data_norm = (data - grav_mean) / grav_std
    data_norm = np.nan_to_num(data_norm, nan=0.0)
    
    inp_tensor = torch.from_numpy(data_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    target_gravity = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
    
    target_mask = ~torch.isnan(target_gravity)
    target_gravity = torch.nan_to_num(target_gravity, nan=0.0)
    
    # Initialize Model & Physics
    model = SinkholePINN(max_density=config.MAX_DENSITY).to(device)
    physics = VoidGravityPhysicsLayer(
        pixel_size,
        config.DEPTH_ESTIMATE,
        thickness=config.THICKNESS_ESTIMATE
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-5)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # Training Loop
    logger.info("Starting Void-Mode Inversion...")
    training_start = time.time()
    
    model.train()
    loop = tqdm(range(config.EPOCHS), desc="Inverting for Voids")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in loop:
        optimizer.zero_grad()
        
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        
        with autocast(device_type, enabled=use_amp):
            # Predict density
            pred_density = model(inp_tensor)
            
            # Physics simulation
            sim_gravity = physics(pred_density)
            
            # Loss calculation
            # A. Data Fidelity (MSE)
            loss_data = F.mse_loss(sim_gravity[target_mask], target_gravity[target_mask])
            
            # B. Sparsity (L1) - Encourage simple models
            loss_sparsity = torch.mean(torch.abs(pred_density))
            
            # C. Total Variation (TV) - Smooth regularization
            diff_i = torch.abs(pred_density[:, :, 1:, :] - pred_density[:, :, :-1, :])
            diff_j = torch.abs(pred_density[:, :, :, 1:] - pred_density[:, :, :, :-1])
            loss_tv = torch.mean(diff_i) + torch.mean(diff_j)
            
            # D. VOID BIAS - Penalize positive mass (we want negative = voids)
            # This is key for sinkhole detection
            loss_void_bias = torch.mean(F.relu(pred_density))  # Penalize positive values
            
            # Weighted Sum
            loss = (config.PHYSICS_WEIGHT * loss_data) + \
                   (config.SPARSITY_WEIGHT * loss_sparsity) + \
                   (config.TV_WEIGHT * loss_tv) + \
                   (config.VOID_BIAS_WEIGHT * loss_void_bias)
        
        # Backpropagation
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            loop.set_postfix(
                loss=loss.item(), 
                mse=loss_data.item(), 
                void_bias=loss_void_bias.item(),
                lr=optimizer.param_groups[0]['lr']
            )
    
    training_time = time.time() - training_start
    logger.info(f"Training completed in {training_time:.1f}s")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Generate output
    model.eval()
    with torch.no_grad():
        final_density = model(inp_tensor).squeeze().cpu().numpy()
    
    # Save result
    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_density.astype(np.float32), 1)
        dst.set_band_description(1, "Density Contrast (kg/m³) - Negative=Void")
    
    # Calculate statistics
    void_pixels = np.sum(final_density < -50)  # Significant negative anomaly
    total_pixels = final_density.size
    void_percentage = (void_pixels / total_pixels) * 100
    
    metadata = {
        'training_time_seconds': training_time,
        'best_loss': best_loss,
        'region': region,
        'region_config': region_config,
        'output_path': output_path,
        'density_stats': {
            'min': float(np.min(final_density)),
            'max': float(np.max(final_density)),
            'mean': float(np.mean(final_density)),
            'std': float(np.std(final_density)),
            'void_pixel_count': int(void_pixels),
            'void_percentage': float(void_percentage)
        }
    }
    
    logger.info(f"Inversion Complete. Void percentage: {void_percentage:.2f}%")
    logger.info(f"Saved to {output_path}")
    
    return metadata


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GeoRisk: Physics-Informed Sinkhole Detection")
    parser.add_argument("--input", required=True, help="Input Bouguer Gravity GeoTIFF")
    parser.add_argument("--output", default="void_density_map.tif", help="Output Density Contrast GeoTIFF")
    parser.add_argument("--region", default="florida_karst", 
                       choices=['florida_karst', 'texas_salt', 'dead_sea', 'konya_basin', 'general'],
                       help="Geological region for threshold calibration")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        config = SinkholeInversionConfig()
        config.EPOCHS = args.epochs
        
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        result = invert_gravity_for_voids(args.input, args.output, config, args.region)
        print(f"\nInversion Results: {result}")
    else:
        logger.error(f"Input file '{args.input}' not found.")
