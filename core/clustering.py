"""
DBSCAN-Based Sinkhole Cluster Detection
========================================

Implements spatial clustering for sinkhole precursor detection using
DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Based on research consensus (2020-2025):
- A true sinkhole precursor affects a CONTINUOUS patch of ground
- Single-pixel anomalies are typically noise (speckle, phase error)
- Clusters of 5-10+ pixels with similar velocities indicate real signals

Key parameters from literature:
- MinPts: 5-10 pixels minimum for valid cluster
- Epsilon: 20-50 meters search radius
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Point, Polygon
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result from cluster detection."""
    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    pixel_count: int
    area_sqm: float
    velocity_mean: float
    velocity_min: float
    velocity_max: float
    velocity_std: float
    density_contrast: Optional[float] = None
    risk_score: float = 0.0
    is_valid: bool = True
    rejection_reason: Optional[str] = None


class SinkholeClusterDetector:
    """
    Detects and validates sinkhole precursor clusters using DBSCAN.
    
    Filters spatially coherent anomalies from noise using:
    1. DBSCAN spatial clustering
    2. Minimum cluster size thresholds
    3. Velocity coherence checks
    4. Morphological analysis (circular vs elongated)
    """
    
    def __init__(self, 
                 min_cluster_size: int = 5,
                 eps_meters: float = 30.0,
                 velocity_coherence_threshold: float = 2.0):
        """
        Initialize the cluster detector.
        
        Args:
            min_cluster_size: Minimum pixels per valid cluster (default 5)
            eps_meters: DBSCAN epsilon in meters (search radius)
            velocity_coherence_threshold: Max std within cluster (mm/yr)
        """
        self.min_cluster_size = min_cluster_size
        self.eps_meters = eps_meters
        self.velocity_coherence = velocity_coherence_threshold
        
        logger.info(f"SinkholeClusterDetector: min_size={min_cluster_size}, "
                   f"eps={eps_meters}m, coherence={velocity_coherence_threshold}")
    
    def detect_clusters_from_raster(self,
                                    velocity_path: str,
                                    velocity_threshold: float = -3.0,
                                    density_path: Optional[str] = None) -> List[ClusterResult]:
        """
        Detect anomaly clusters from velocity raster.
        
        Args:
            velocity_path: Path to velocity GeoTIFF (mm/yr)
            velocity_threshold: Velocity below which to cluster (e.g., -3 mm/yr)
            density_path: Optional density contrast map for additional scoring
            
        Returns:
            List of ClusterResult objects for valid clusters
        """
        with rasterio.open(velocity_path) as src:
            velocity = src.read(1)
            transform = src.transform
            crs = src.crs
            pixel_size_x = transform[0]
            pixel_size_y = -transform[4]  # Negative for north-up
            
            # Convert to meters if geographic
            if src.crs and src.crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2
                deg_to_m = 111320 * np.cos(np.deg2rad(center_lat))
                pixel_size_m = pixel_size_x * deg_to_m
                eps_pixels = self.eps_meters / pixel_size_m
            else:
                pixel_size_m = pixel_size_x
                eps_pixels = self.eps_meters / pixel_size_m
        
        # Load density if provided
        density = None
        if density_path:
            with rasterio.open(density_path) as src2:
                density = src2.read(1)
        
        # Create anomaly mask
        anomaly_mask = velocity < velocity_threshold
        anomaly_mask = np.nan_to_num(anomaly_mask, nan=False).astype(bool)
        
        # Get coordinates of anomaly pixels
        rows, cols = np.where(anomaly_mask)
        
        if len(rows) == 0:
            logger.info("No pixels below velocity threshold")
            return []
        
        # Prepare data for DBSCAN
        # Scale coordinates to approximately equal spacing
        coords = np.column_stack([rows, cols])
        velocities = velocity[rows, cols]
        
        # DBSCAN clustering
        logger.info(f"Running DBSCAN on {len(coords)} anomaly pixels...")
        db = DBSCAN(eps=eps_pixels, min_samples=self.min_cluster_size)
        cluster_labels = db.fit_predict(coords)
        
        # Extract unique clusters (ignore noise label -1)
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)
        
        logger.info(f"Found {len(unique_labels)} clusters (noise points: {np.sum(cluster_labels == -1)})")
        
        # Analyze each cluster
        clusters = []
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_rows = rows[mask]
            cluster_cols = cols[mask]
            cluster_velocities = velocities[mask]
            
            # Basic statistics
            pixel_count = len(cluster_rows)
            velocity_mean = np.mean(cluster_velocities)
            velocity_min = np.min(cluster_velocities)
            velocity_max = np.max(cluster_velocities)
            velocity_std = np.std(cluster_velocities)
            
            # Calculate centroid
            centroid_row = np.mean(cluster_rows)
            centroid_col = np.mean(cluster_cols)
            lon, lat = rasterio.transform.xy(transform, centroid_row, centroid_col)
            
            # Calculate area
            area_sqm = pixel_count * (pixel_size_m ** 2)
            
            # Get density if available
            density_contrast = None
            if density is not None:
                density_values = density[cluster_rows, cluster_cols]
                density_contrast = np.mean(density_values)
            
            # Validation checks
            is_valid = True
            rejection_reason = None
            
            # Check velocity coherence
            if velocity_std > self.velocity_coherence * abs(velocity_mean):
                is_valid = False
                rejection_reason = f"Low coherence (std={velocity_std:.2f})"
            
            # Check minimum size (redundant with DBSCAN but explicit)
            if pixel_count < self.min_cluster_size:
                is_valid = False
                rejection_reason = f"Too small ({pixel_count} pixels)"
            
            # Calculate initial risk score
            risk_score = self._calculate_cluster_risk_score(
                velocity_mean, velocity_min, velocity_std, 
                pixel_count, area_sqm, density_contrast
            )
            
            cluster = ClusterResult(
                cluster_id=int(label),
                centroid_lat=lat,
                centroid_lon=lon,
                pixel_count=pixel_count,
                area_sqm=area_sqm,
                velocity_mean=velocity_mean,
                velocity_min=velocity_min,
                velocity_max=velocity_max,
                velocity_std=velocity_std,
                density_contrast=density_contrast,
                risk_score=risk_score,
                is_valid=is_valid,
                rejection_reason=rejection_reason
            )
            
            clusters.append(cluster)
        
        # Sort by risk score
        clusters.sort(key=lambda x: x.risk_score, reverse=True)
        
        valid_count = sum(1 for c in clusters if c.is_valid)
        logger.info(f"Valid clusters: {valid_count}/{len(clusters)}")
        
        return clusters
    
    def _calculate_cluster_risk_score(self,
                                     velocity_mean: float,
                                     velocity_min: float,
                                     velocity_std: float,
                                     pixel_count: int,
                                     area_sqm: float,
                                     density_contrast: Optional[float]) -> float:
        """
        Calculate a composite risk score for a cluster.
        
        Score factors:
        - Velocity magnitude (more negative = higher risk)
        - Cluster size (larger = higher confidence)
        - Velocity coherence (low std = higher confidence)
        - Density contrast (more negative = higher risk, if available)
        """
        score = 0.0
        
        # Velocity component (0-40 points)
        # -3 mm/yr = 10 points, -10 mm/yr = 40 points
        velocity_score = min(40, max(0, (-velocity_mean - 3) * 4.3))
        score += velocity_score
        
        # Size component (0-20 points)
        # 5 pixels = 2 points, 50 pixels = 20 points
        size_score = min(20, max(0, pixel_count * 0.4))
        score += size_score
        
        # Coherence component (0-20 points)
        # Low std = high score
        if velocity_std > 0:
            coherence_ratio = abs(velocity_mean) / velocity_std
            coherence_score = min(20, max(0, coherence_ratio * 4))
        else:
            coherence_score = 20
        score += coherence_score
        
        # Density component (0-20 points, if available)
        if density_contrast is not None:
            # More negative density = more void = higher risk
            density_score = min(20, max(0, (-density_contrast - 50) * 0.1))
            score += density_score
        
        return min(100, score)
    
    def clusters_to_geodataframe(self, clusters: List[ClusterResult]) -> gpd.GeoDataFrame:
        """Convert cluster list to GeoDataFrame for GIS export."""
        records = []
        for c in clusters:
            records.append({
                'cluster_id': c.cluster_id,
                'geometry': Point(c.centroid_lon, c.centroid_lat),
                'pixel_count': c.pixel_count,
                'area_sqm': c.area_sqm,
                'velocity_mean': c.velocity_mean,
                'velocity_min': c.velocity_min,
                'velocity_max': c.velocity_max,
                'velocity_std': c.velocity_std,
                'density_contrast': c.density_contrast,
                'risk_score': c.risk_score,
                'is_valid': c.is_valid,
                'rejection_reason': c.rejection_reason
            })
        
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        return gdf
    
    def clusters_to_dataframe(self, clusters: List[ClusterResult]) -> pd.DataFrame:
        """Convert cluster list to pandas DataFrame."""
        records = []
        for c in clusters:
            records.append({
                'cluster_id': c.cluster_id,
                'latitude': c.centroid_lat,
                'longitude': c.centroid_lon,
                'pixel_count': c.pixel_count,
                'area_sqm': c.area_sqm,
                'velocity_mean': c.velocity_mean,
                'velocity_min': c.velocity_min,
                'velocity_max': c.velocity_max,
                'velocity_std': c.velocity_std,
                'density_contrast': c.density_contrast,
                'risk_score': c.risk_score,
                'is_valid': c.is_valid,
                'rejection_reason': c.rejection_reason
            })
        
        return pd.DataFrame(records)


class MorphologicalAnalyzer:
    """
    Analyzes cluster morphology to distinguish sinkhole signatures.
    
    Sinkholes typically have:
    - Circular/elliptical shape ("bullseye" pattern)
    - Concentric velocity gradients
    - Distinct boundary with surrounding stable terrain
    """
    
    @staticmethod
    def calculate_circularity(pixel_coords: np.ndarray) -> float:
        """
        Calculate circularity metric (1.0 = perfect circle).
        
        Circularity = 4π * Area / Perimeter²
        """
        from scipy.spatial import ConvexHull
        
        if len(pixel_coords) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(pixel_coords)
            area = hull.volume  # In 2D, "volume" is area
            # Approximate perimeter from hull edge lengths
            perimeter = 0
            for i in range(len(hull.vertices)):
                p1 = pixel_coords[hull.vertices[i]]
                p2 = pixel_coords[hull.vertices[(i+1) % len(hull.vertices)]]
                perimeter += np.sqrt(np.sum((p1 - p2) ** 2))
            
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                return min(1.0, circularity)
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_aspect_ratio(pixel_coords: np.ndarray) -> float:
        """
        Calculate aspect ratio (major/minor axis of fitted ellipse).
        
        Sinkholes typically have aspect ratio < 2.0
        Elongated features (infrastructure settlement) have higher ratios
        """
        if len(pixel_coords) < 3:
            return 1.0
        
        # Use PCA to find principal axes
        centered = pixel_coords - np.mean(pixel_coords, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            if eigenvalues[1] > 0:
                aspect_ratio = np.sqrt(eigenvalues[0] / eigenvalues[1])
                return aspect_ratio
            return 1.0
        except:
            return 1.0
    
    @staticmethod
    def is_sinkhole_morphology(pixel_coords: np.ndarray,
                               velocities: np.ndarray,
                               circularity_threshold: float = 0.5,
                               aspect_ratio_threshold: float = 3.0) -> Tuple[bool, Dict]:
        """
        Determine if cluster morphology is consistent with sinkhole.
        
        Returns (is_sinkhole, metrics_dict)
        """
        circularity = MorphologicalAnalyzer.calculate_circularity(pixel_coords)
        aspect_ratio = MorphologicalAnalyzer.calculate_aspect_ratio(pixel_coords)
        
        # Check for concentric gradient (center should be more negative)
        centroid = np.mean(pixel_coords, axis=0)
        distances = np.sqrt(np.sum((pixel_coords - centroid) ** 2, axis=1))
        
        # Correlation between distance and velocity
        # For sinkholes: center (low distance) should have more negative velocity
        correlation = np.corrcoef(distances, velocities)[0, 1]
        has_gradient = correlation > 0.2  # Positive correlation = center more negative
        
        is_sinkhole = (
            circularity >= circularity_threshold and
            aspect_ratio <= aspect_ratio_threshold
        )
        
        return is_sinkhole, {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'has_concentric_gradient': has_gradient,
            'gradient_correlation': correlation
        }


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GeoRisk: Sinkhole Cluster Detection")
    parser.add_argument("--velocity", required=True, help="Input Velocity GeoTIFF (mm/yr)")
    parser.add_argument("--density", help="Optional Density Contrast GeoTIFF")
    parser.add_argument("--threshold", type=float, default=-3.0, help="Velocity threshold (mm/yr)")
    parser.add_argument("--min-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--eps", type=float, default=30.0, help="DBSCAN epsilon (meters)")
    parser.add_argument("--output", default="clusters.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    import os
    if os.path.exists(args.velocity):
        detector = SinkholeClusterDetector(
            min_cluster_size=args.min_size,
            eps_meters=args.eps
        )
        
        clusters = detector.detect_clusters_from_raster(
            args.velocity,
            velocity_threshold=args.threshold,
            density_path=args.density
        )
        
        if clusters:
            df = detector.clusters_to_dataframe(clusters)
            df.to_csv(args.output, index=False)
            print(f"\nSaved {len(clusters)} clusters to {args.output}")
            
            valid_clusters = [c for c in clusters if c.is_valid]
            print(f"\nTop 10 Valid Clusters by Risk Score:")
            for c in valid_clusters[:10]:
                print(f"  ID={c.cluster_id}: Score={c.risk_score:.1f}, "
                      f"Vel={c.velocity_mean:.1f} mm/yr, "
                      f"Area={c.area_sqm:.0f} m², "
                      f"Lat={c.centroid_lat:.5f}, Lon={c.centroid_lon:.5f}")
        else:
            print("No clusters detected.")
    else:
        logger.error(f"Input file '{args.velocity}' not found.")
