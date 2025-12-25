"""
InSAR Velocity Analysis Module
==============================

Processes InSAR (Interferometric Synthetic Aperture Radar) time-series data
to extract ground deformation velocities for sinkhole precursor detection.

Based on research consensus (2020-2025) that identifies:
- "Golden Window" precursor detection periods
- Region-specific velocity thresholds
- Acceleration analysis for acute vs chronic differentiation

Key velocity thresholds (from literature):
- Florida Karst: -3 to -6 mm/yr (Warning to Critical)
- Salt Domes: -10 to -20 mm/yr
- Evaporite: -15 to -30 mm/yr
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from scipy.stats import linregress
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification levels for insurance purposes."""
    STABLE = "stable"
    MONITORING = "monitoring"
    WARNING = "warning"
    CRITICAL = "critical"
    IMMINENT = "imminent"


@dataclass
class VelocityThresholds:
    """Region-specific velocity thresholds based on literature consensus."""
    region: str
    background_noise: float = -2.0      # mm/yr - below this is noise
    warning_threshold: float = -3.0     # mm/yr - Yellow flag
    critical_threshold: float = -6.0    # mm/yr - Red flag
    imminent_threshold: float = -12.0   # mm/yr - Immediate risk
    acceleration_threshold: float = 0.5 # 50% velocity increase = acute
    sensor_type: str = "X-band"
    description: str = ""


# Pre-configured regional thresholds from research
REGIONAL_THRESHOLDS = {
    'florida_karst': VelocityThresholds(
        region='florida_karst',
        background_noise=-2.0,
        warning_threshold=-3.0,
        critical_threshold=-6.0,
        imminent_threshold=-10.0,
        sensor_type='X-band',
        description='Cover-collapse sinkholes in sand/limestone mantle'
    ),
    'texas_salt': VelocityThresholds(
        region='texas_salt',
        background_noise=-5.0,
        warning_threshold=-10.0,
        critical_threshold=-20.0,
        imminent_threshold=-30.0,
        sensor_type='C-band',
        description='Salt dome dissolution with massive horizontal strain'
    ),
    'dead_sea': VelocityThresholds(
        region='dead_sea',
        background_noise=-5.0,
        warning_threshold=-15.0,
        critical_threshold=-20.0,
        imminent_threshold=-30.0,
        sensor_type='C-band',
        description='Evaporite dissolution with viscoelastic overburden'
    ),
    'konya_basin': VelocityThresholds(
        region='konya_basin',
        background_noise=-5.0,
        warning_threshold=-15.0,
        critical_threshold=-30.0,
        imminent_threshold=-50.0,
        sensor_type='C-band',
        description='Anthropogenic groundwater depletion sinkholes'
    ),
    'general': VelocityThresholds(
        region='general',
        background_noise=-2.0,
        warning_threshold=-5.0,
        critical_threshold=-10.0,
        imminent_threshold=-20.0,
        sensor_type='C-band',
        description='Generic karst terrain'
    )
}


@dataclass
class VelocityMeasurement:
    """Single point velocity measurement with metadata."""
    latitude: float
    longitude: float
    velocity_vertical: float          # mm/yr - Line of Sight or decomposed vertical
    velocity_horizontal_ew: Optional[float] = None  # mm/yr - East-West component
    velocity_horizontal_ns: Optional[float] = None  # mm/yr - North-South component
    velocity_los: Optional[float] = None  # mm/yr - Original Line of Sight
    coherence: float = 1.0
    timestamp: Optional[str] = None
    point_id: Optional[str] = None
    

@dataclass
class TimeSeriesPoint:
    """Multi-temporal observation point for acceleration analysis."""
    point_id: str
    latitude: float
    longitude: float
    dates: List[str] = field(default_factory=list)
    displacements: List[float] = field(default_factory=list)  # cumulative mm
    velocities: List[float] = field(default_factory=list)     # mm/yr at each epoch
    coherences: List[float] = field(default_factory=list)
    

class VelocityAnalyzer:
    """
    Analyzes InSAR velocity data for sinkhole precursor detection.
    
    Implements:
    - Velocity classification against regional thresholds
    - Acceleration detection (exponential vs logarithmic trends)
    - Horizontal strain analysis for salt dome regions
    """
    
    def __init__(self, region: str = 'general'):
        self.thresholds = REGIONAL_THRESHOLDS.get(region, REGIONAL_THRESHOLDS['general'])
        self.region = region
        logger.info(f"VelocityAnalyzer initialized for region: {region}")
        logger.info(f"Thresholds: Warning={self.thresholds.warning_threshold}, "
                   f"Critical={self.thresholds.critical_threshold}")
    
    def classify_velocity(self, velocity: float) -> RiskLevel:
        """
        Classify a single velocity value to a risk level.
        
        Note: Velocities are negative (subsidence), so we use absolute value logic.
        More negative = higher risk.
        """
        if velocity > self.thresholds.background_noise:
            return RiskLevel.STABLE
        elif velocity > self.thresholds.warning_threshold:
            return RiskLevel.MONITORING
        elif velocity > self.thresholds.critical_threshold:
            return RiskLevel.WARNING
        elif velocity > self.thresholds.imminent_threshold:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.IMMINENT
    
    def analyze_acceleration(self, time_series: TimeSeriesPoint, 
                           rolling_window_months: int = 6) -> Dict:
        """
        Analyze velocity time series to detect acceleration patterns.
        
        Differentiates:
        - Logarithmic decay (benign soil consolidation) - settling
        - Exponential acceleration (void propagation) - DANGER
        
        Returns acceleration metrics and pattern classification.
        """
        if len(time_series.displacements) < 3:
            return {
                'pattern': 'insufficient_data',
                'acceleration_rate': 0.0,
                'is_accelerating': False,
                'confidence': 0.0
            }
        
        displacements = np.array(time_series.displacements)
        times = np.arange(len(displacements))
        
        # Calculate instantaneous velocities (diff)
        velocities = np.diff(displacements)
        velocity_times = times[1:]
        
        if len(velocities) < 2:
            return {
                'pattern': 'insufficient_data',
                'acceleration_rate': 0.0,
                'is_accelerating': False,
                'confidence': 0.0
            }
        
        # Linear regression on velocities to detect trend
        slope, intercept, r_value, p_value, std_err = linregress(velocity_times, velocities)
        
        # Calculate acceleration as percentage change
        if len(velocities) >= 2 and velocities[0] != 0:
            recent_velocity = np.mean(velocities[-3:]) if len(velocities) >= 3 else velocities[-1]
            early_velocity = np.mean(velocities[:3]) if len(velocities) >= 3 else velocities[0]
            
            if early_velocity != 0:
                acceleration_pct = (recent_velocity - early_velocity) / abs(early_velocity)
            else:
                acceleration_pct = 0.0
        else:
            acceleration_pct = 0.0
        
        # Pattern classification
        is_accelerating = slope < -0.1 and acceleration_pct > 0.5  # Getting MORE negative
        
        if is_accelerating:
            pattern = 'exponential_acceleration'  # DANGER - void propagation
        elif slope > 0.1:
            pattern = 'logarithmic_decay'  # Benign - soil consolidation
        else:
            pattern = 'linear_stable'
        
        return {
            'pattern': pattern,
            'acceleration_rate': float(slope),
            'acceleration_pct': float(acceleration_pct),
            'is_accelerating': is_accelerating,
            'r_squared': float(r_value ** 2),
            'confidence': float(abs(r_value)),
            'risk_multiplier': 2.0 if is_accelerating else 1.0
        }
    
    def analyze_horizontal_strain(self, 
                                  ascending_los: np.ndarray,
                                  descending_los: np.ndarray,
                                  incidence_angle: float = 39.0) -> Dict:
        """
        Decompose ascending/descending LOS data to extract horizontal strain.
        
        Critical for salt dome regions (Bayou Corne) where horizontal precursors
        (260mm lateral shift) may precede vertical collapse.
        
        Args:
            ascending_los: LOS velocity from ascending pass (East-looking)
            descending_los: LOS velocity from descending pass (West-looking)
            incidence_angle: SAR incidence angle in degrees
            
        Returns:
            Decomposed vertical and East-West components
        """
        # Convert incidence angle to radians
        theta = np.deg2rad(incidence_angle)
        
        # LOS decomposition (simplified 2D case)
        # d_LOS_asc = d_v * cos(theta) + d_ew * sin(theta)  (for right-looking SAR)
        # d_LOS_desc = d_v * cos(theta) - d_ew * sin(theta)
        
        # Solve for vertical and EW
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Add + equation: 2 * d_v * cos(theta) = d_asc + d_desc
        vertical = (ascending_los + descending_los) / (2 * cos_theta)
        
        # Subtract equation: 2 * d_ew * sin(theta) = d_asc - d_desc
        horizontal_ew = (ascending_los - descending_los) / (2 * sin_theta)
        
        # Calculate strain gradient (horizontal gradient in EW direction)
        ew_gradient = np.gradient(horizontal_ew)
        strain_magnitude = np.abs(ew_gradient)
        
        return {
            'vertical_velocity': vertical,
            'horizontal_ew_velocity': horizontal_ew,
            'strain_gradient': ew_gradient,
            'max_strain': float(np.max(strain_magnitude)),
            'mean_strain': float(np.mean(strain_magnitude)),
            'high_strain_threshold': 0.01,  # 1% strain is significant
            'high_strain_pixels': int(np.sum(strain_magnitude > 0.01))
        }


class InSARProcessor:
    """
    Processes InSAR data for insurance-grade sinkhole risk assessment.
    
    Supports:
    - Velocity raster ingestion (GeoTIFF)
    - Point cloud (PS-InSAR) processing
    - Multi-pass decomposition
    - Regional threshold application
    """
    
    def __init__(self, region: str = 'general'):
        self.region = region
        self.analyzer = VelocityAnalyzer(region)
        self.thresholds = REGIONAL_THRESHOLDS.get(region, REGIONAL_THRESHOLDS['general'])
    
    def process_velocity_raster(self, 
                                velocity_tif_path: str,
                                output_risk_path: Optional[str] = None) -> Dict:
        """
        Process a velocity raster (velocity map from InSAR processing).
        
        Args:
            velocity_tif_path: Path to velocity GeoTIFF (units: mm/yr, negative=subsidence)
            output_risk_path: Optional path for risk classification raster
            
        Returns:
            Dictionary with statistics and risk classification counts
        """
        with rasterio.open(velocity_tif_path) as src:
            velocity = src.read(1)
            profile = src.profile
            transform = src.transform
            nodata = src.nodata
            
        # Handle nodata
        if nodata is not None:
            velocity = np.where(velocity == nodata, np.nan, velocity)
        
        valid_mask = ~np.isnan(velocity)
        valid_velocity = velocity[valid_mask]
        
        # Classify each pixel
        risk_map = np.full_like(velocity, np.nan, dtype=np.float32)
        
        # Risk levels as numeric: 0=stable, 1=monitoring, 2=warning, 3=critical, 4=imminent
        risk_levels = {
            RiskLevel.STABLE: 0,
            RiskLevel.MONITORING: 1,
            RiskLevel.WARNING: 2,
            RiskLevel.CRITICAL: 3,
            RiskLevel.IMMINENT: 4
        }
        
        for i in range(velocity.shape[0]):
            for j in range(velocity.shape[1]):
                if valid_mask[i, j]:
                    level = self.analyzer.classify_velocity(velocity[i, j])
                    risk_map[i, j] = risk_levels[level]
        
        # Calculate statistics
        stats = {
            'region': self.region,
            'thresholds': {
                'warning': self.thresholds.warning_threshold,
                'critical': self.thresholds.critical_threshold,
                'imminent': self.thresholds.imminent_threshold
            },
            'velocity_stats': {
                'min': float(np.nanmin(velocity)),
                'max': float(np.nanmax(velocity)),
                'mean': float(np.nanmean(velocity)),
                'std': float(np.nanstd(velocity)),
                'median': float(np.nanmedian(velocity))
            },
            'risk_classification': {
                'stable_pixels': int(np.sum(risk_map == 0)),
                'monitoring_pixels': int(np.sum(risk_map == 1)),
                'warning_pixels': int(np.sum(risk_map == 2)),
                'critical_pixels': int(np.sum(risk_map == 3)),
                'imminent_pixels': int(np.sum(risk_map == 4)),
                'total_valid_pixels': int(np.sum(valid_mask))
            }
        }
        
        # Calculate percentages
        total = stats['risk_classification']['total_valid_pixels']
        if total > 0:
            stats['risk_percentages'] = {
                'warning_pct': stats['risk_classification']['warning_pixels'] / total * 100,
                'critical_pct': stats['risk_classification']['critical_pixels'] / total * 100,
                'imminent_pct': stats['risk_classification']['imminent_pixels'] / total * 100
            }
        
        # Save risk map if path provided
        if output_risk_path:
            profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            with rasterio.open(output_risk_path, 'w', **profile) as dst:
                dst.write(risk_map.astype(np.float32), 1)
                dst.set_band_description(1, "Risk Level (0=stable...4=imminent)")
            logger.info(f"Risk map saved to {output_risk_path}")
        
        logger.info(f"Processed velocity raster: {stats['risk_classification']}")
        
        return stats
    
    def extract_warning_zones(self, 
                             velocity_tif_path: str,
                             min_cluster_size: int = 5) -> pd.DataFrame:
        """
        Extract zones at warning level or above for insurance reporting.
        
        Uses connected component analysis to identify contiguous warning areas.
        
        Returns DataFrame with zone centroids, max velocities, and areas.
        """
        from scipy.ndimage import label, find_objects
        
        with rasterio.open(velocity_tif_path) as src:
            velocity = src.read(1)
            transform = src.transform
            crs = src.crs
            
        # Create warning mask (velocity below warning threshold)
        warning_mask = velocity < self.thresholds.warning_threshold
        warning_mask = np.nan_to_num(warning_mask, nan=False).astype(bool)
        
        # Connected components
        labeled_array, num_features = label(warning_mask)
        objects = find_objects(labeled_array)
        
        zones = []
        for i, sl in enumerate(objects, start=1):
            if sl is None:
                continue
                
            component_mask = (labeled_array[sl] == i)
            component_velocity = velocity[sl]
            
            area_pixels = np.sum(component_mask)
            if area_pixels < min_cluster_size:
                continue
            
            # Get centroid
            rows, cols = np.where(component_mask)
            centroid_row = sl[0].start + np.mean(rows)
            centroid_col = sl[1].start + np.mean(cols)
            
            # Convert to coordinates
            lon, lat = rasterio.transform.xy(transform, centroid_row, centroid_col)
            
            # Get velocity statistics for zone
            zone_velocities = component_velocity[component_mask]
            
            zones.append({
                'zone_id': i,
                'latitude': lat,
                'longitude': lon,
                'area_pixels': int(area_pixels),
                'velocity_min': float(np.min(zone_velocities)),
                'velocity_max': float(np.max(zone_velocities)),
                'velocity_mean': float(np.mean(zone_velocities)),
                'risk_level': self.analyzer.classify_velocity(np.min(zone_velocities)).value
            })
        
        df = pd.DataFrame(zones)
        if not df.empty:
            df = df.sort_values('velocity_min').reset_index(drop=True)
            
        logger.info(f"Extracted {len(zones)} warning zones")
        
        return df


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GeoRisk: InSAR Velocity Analysis")
    parser.add_argument("--input", required=True, help="Input Velocity GeoTIFF (mm/yr)")
    parser.add_argument("--output", default="risk_classification.tif", help="Output Risk Map")
    parser.add_argument("--region", default="general",
                       choices=list(REGIONAL_THRESHOLDS.keys()),
                       help="Geological region for thresholds")
    parser.add_argument("--extract-zones", action="store_true", help="Extract warning zones to CSV")
    
    args = parser.parse_args()
    
    import os
    if os.path.exists(args.input):
        processor = InSARProcessor(region=args.region)
        
        stats = processor.process_velocity_raster(args.input, args.output)
        print(f"\nVelocity Stats: {stats['velocity_stats']}")
        print(f"Risk Classification: {stats['risk_classification']}")
        
        if args.extract_zones:
            zones_df = processor.extract_warning_zones(args.input)
            zones_csv = args.output.replace('.tif', '_zones.csv')
            zones_df.to_csv(zones_csv, index=False)
            print(f"\nWarning zones saved to {zones_csv}")
    else:
        logger.error(f"Input file '{args.input}' not found.")
