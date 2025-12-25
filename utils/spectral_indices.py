"""
Spectral Index Calculations for Sinkhole Precursor Detection
=============================================================

Implements vegetation and moisture indices for early warning detection:
- NDVI (Normalized Difference Vegetation Index): Vegetation stress indicator
- MI (Moisture Index): Soil/surface moisture anomaly detector

Based on research findings (Poland 2024) detecting spectral "structural breaks"
6 months before sinkhole formation.

Key signals:
- NDVI drop: Root shear from ground movement
- NDVI spike: Water pooling in subsidence bowl
- MI anomaly: Groundwater rebound or surface water accumulation
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import zscore
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SpectralAnomaly:
    """Detected spectral anomaly location."""
    latitude: float
    longitude: float
    index_name: str
    anomaly_type: str  # 'high', 'low', 'variance'
    value: float
    z_score: float
    pixel_count: int = 1


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Range: -1 to +1
    - Dense vegetation: 0.6 to 0.9
    - Sparse vegetation: 0.2 to 0.5
    - Bare soil: -0.1 to 0.2
    - Water: -0.3 to 0
    
    Args:
        nir: Near-infrared band (Sentinel-2 B8)
        red: Red band (Sentinel-2 B4)
        
    Returns:
        NDVI array (-1 to 1)
    """
    # Avoid division by zero
    denominator = nir.astype(float) + red.astype(float)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    ndvi = (nir.astype(float) - red.astype(float)) / denominator
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi


def calculate_moisture_index(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate Moisture Index.
    
    MI = (NIR - SWIR) / (NIR + SWIR)
    
    Higher values indicate more moisture.
    Used to detect water accumulation in subsidence bowls.
    
    Args:
        nir: Near-infrared band (Sentinel-2 B8)
        swir: Short-wave infrared band (Sentinel-2 B11 or B12)
        
    Returns:
        Moisture Index array
    """
    denominator = nir.astype(float) + swir.astype(float)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    mi = (nir.astype(float) - swir.astype(float)) / denominator
    
    return mi


def calculate_local_variance(arr: np.ndarray, window_size: int = 11) -> np.ndarray:
    """
    Calculate local variance to detect anomalous patches.
    
    High localized variance in otherwise uniform area indicates
    vegetation stress or moisture change from subsurface activity.
    
    Args:
        arr: Input spectral index array
        window_size: Size of analysis window
        
    Returns:
        Local variance array
    """
    # Calculate local mean
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(arr, kernel, mode='reflect')
    
    # Calculate local variance
    local_var = ndimage.convolve((arr - local_mean) ** 2, kernel, mode='reflect')
    
    return local_var


def detect_spectral_anomalies(index_array: np.ndarray,
                              transform: rasterio.Affine,
                              index_name: str = 'NDVI',
                              z_threshold: float = 2.5,
                              min_cluster_size: int = 3) -> List[SpectralAnomaly]:
    """
    Detect statistically significant spectral anomalies.
    
    Identifies pixels that deviate significantly from local context,
    which may indicate vegetation stress or moisture changes from
    subsurface void propagation.
    
    Args:
        index_array: Spectral index array (NDVI, MI, etc.)
        transform: Geotransform for coordinate conversion
        index_name: Name of the index for labeling
        z_threshold: Z-score threshold for anomaly detection
        min_cluster_size: Minimum pixels for valid anomaly
        
    Returns:
        List of SpectralAnomaly objects
    """
    # Handle NaN values
    valid_mask = ~np.isnan(index_array)
    if not np.any(valid_mask):
        return []
    
    # Calculate local statistics
    local_var = calculate_local_variance(np.nan_to_num(index_array, nan=0))
    
    # Z-score normalization
    flat_values = index_array[valid_mask]
    mean_val = np.nanmean(flat_values)
    std_val = np.nanstd(flat_values)
    
    if std_val < 1e-10:
        return []
    
    z_scores = (index_array - mean_val) / std_val
    
    # Identify high and low anomalies
    high_anomaly_mask = (z_scores > z_threshold) & valid_mask
    low_anomaly_mask = (z_scores < -z_threshold) & valid_mask
    
    # Also identify high variance zones
    var_threshold = np.nanpercentile(local_var[valid_mask], 95)
    high_var_mask = local_var > var_threshold
    
    anomalies = []
    
    # Process high anomalies (could be water pooling causing vegetation spike)
    from scipy.ndimage import label, find_objects
    labeled_high, num_high = label(high_anomaly_mask)
    for i in range(1, num_high + 1):
        mask = labeled_high == i
        count = np.sum(mask)
        if count >= min_cluster_size:
            rows, cols = np.where(mask)
            centroid_row = np.mean(rows)
            centroid_col = np.mean(cols)
            lon, lat = rasterio.transform.xy(transform, centroid_row, centroid_col)
            
            mean_z = np.mean(z_scores[mask])
            mean_val = np.mean(index_array[mask])
            
            anomalies.append(SpectralAnomaly(
                latitude=lat,
                longitude=lon,
                index_name=index_name,
                anomaly_type='high',
                value=float(mean_val),
                z_score=float(mean_z),
                pixel_count=int(count)
            ))
    
    # Process low anomalies (could be vegetation stress from root shear)
    labeled_low, num_low = label(low_anomaly_mask)
    for i in range(1, num_low + 1):
        mask = labeled_low == i
        count = np.sum(mask)
        if count >= min_cluster_size:
            rows, cols = np.where(mask)
            centroid_row = np.mean(rows)
            centroid_col = np.mean(cols)
            lon, lat = rasterio.transform.xy(transform, centroid_row, centroid_col)
            
            mean_z = np.mean(z_scores[mask])
            mean_val = np.mean(index_array[mask])
            
            anomalies.append(SpectralAnomaly(
                latitude=lat,
                longitude=lon,
                index_name=index_name,
                anomaly_type='low',
                value=float(mean_val),
                z_score=float(mean_z),
                pixel_count=int(count)
            ))
    
    logger.info(f"Detected {len(anomalies)} {index_name} anomalies")
    
    return anomalies


def calculate_temporal_change(current: np.ndarray,
                              baseline: np.ndarray,
                              change_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate temporal change between two spectral index images.
    
    Structural breaks in time series (sudden changes) may indicate
    onset of sinkhole formation (6 months lead time per literature).
    
    Args:
        current: Current spectral index array
        baseline: Baseline (earlier) spectral index array
        change_threshold: Minimum absolute change to flag
        
    Returns:
        Tuple of (change array, significant_change_mask)
    """
    change = current - baseline
    
    significant_mask = np.abs(change) > change_threshold
    
    return change, significant_mask


def process_sentinel2_for_sinkhole(nir_path: str,
                                   red_path: str,
                                   swir_path: str,
                                   output_dir: str) -> Dict[str, str]:
    """
    Process Sentinel-2 bands to generate sinkhole-relevant spectral indices.
    
    Args:
        nir_path: Path to NIR band (B8)
        red_path: Path to Red band (B4)
        swir_path: Path to SWIR band (B11 or B12)
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of output paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load bands
    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(float)
        profile = src.profile
        transform = src.transform
    
    with rasterio.open(red_path) as src:
        red = src.read(1).astype(float)
    
    with rasterio.open(swir_path) as src:
        swir = src.read(1).astype(float)
    
    outputs = {}
    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    
    # Calculate NDVI
    logger.info("Calculating NDVI...")
    ndvi = calculate_ndvi(nir, red)
    ndvi_path = os.path.join(output_dir, 'ndvi.tif')
    with rasterio.open(ndvi_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)
        dst.set_band_description(1, "NDVI")
    outputs['ndvi'] = ndvi_path
    
    # Calculate Moisture Index
    logger.info("Calculating Moisture Index...")
    mi = calculate_moisture_index(nir, swir)
    mi_path = os.path.join(output_dir, 'moisture_index.tif')
    with rasterio.open(mi_path, 'w', **profile) as dst:
        dst.write(mi.astype(np.float32), 1)
        dst.set_band_description(1, "Moisture Index")
    outputs['moisture_index'] = mi_path
    
    # Calculate variance maps
    logger.info("Calculating variance maps...")
    ndvi_var = calculate_local_variance(ndvi)
    ndvi_var_path = os.path.join(output_dir, 'ndvi_variance.tif')
    with rasterio.open(ndvi_var_path, 'w', **profile) as dst:
        dst.write(ndvi_var.astype(np.float32), 1)
        dst.set_band_description(1, "NDVI Local Variance")
    outputs['ndvi_variance'] = ndvi_var_path
    
    mi_var = calculate_local_variance(mi)
    mi_var_path = os.path.join(output_dir, 'mi_variance.tif')
    with rasterio.open(mi_var_path, 'w', **profile) as dst:
        dst.write(mi_var.astype(np.float32), 1)
        dst.set_band_description(1, "Moisture Index Local Variance")
    outputs['mi_variance'] = mi_var_path
    
    # Detect anomalies
    logger.info("Detecting spectral anomalies...")
    ndvi_anomalies = detect_spectral_anomalies(ndvi, transform, 'NDVI')
    mi_anomalies = detect_spectral_anomalies(mi, transform, 'MI')
    
    # Save anomalies to CSV
    import pandas as pd
    all_anomalies = ndvi_anomalies + mi_anomalies
    if all_anomalies:
        df = pd.DataFrame([{
            'latitude': a.latitude,
            'longitude': a.longitude,
            'index': a.index_name,
            'type': a.anomaly_type,
            'value': a.value,
            'z_score': a.z_score,
            'pixel_count': a.pixel_count
        } for a in all_anomalies])
        anomalies_path = os.path.join(output_dir, 'spectral_anomalies.csv')
        df.to_csv(anomalies_path, index=False)
        outputs['anomalies'] = anomalies_path
    
    logger.info(f"Saved spectral outputs to {output_dir}")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate spectral indices for sinkhole detection")
    parser.add_argument("--nir", required=True, help="NIR band GeoTIFF (Sentinel-2 B8)")
    parser.add_argument("--red", required=True, help="Red band GeoTIFF (Sentinel-2 B4)")
    parser.add_argument("--swir", required=True, help="SWIR band GeoTIFF (Sentinel-2 B11)")
    parser.add_argument("--output-dir", default="spectral_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    import os
    if all(os.path.exists(p) for p in [args.nir, args.red, args.swir]):
        outputs = process_sentinel2_for_sinkhole(
            args.nir, args.red, args.swir, args.output_dir
        )
        print(f"\nGenerated outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    else:
        logger.error("One or more input files not found.")
