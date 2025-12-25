"""
Geological Proxy Calculations
=============================

Implements terrain-derived proxies for sinkhole susceptibility:
- Convergence Index (CI): Flow concentration metric
- Topographic Wetness Index (TWI): Drainage/saturation indicator
- Terrain Ruggedness Index (TRI): Surface roughness detector
- Plan Curvature: Depression/bowl identification

Based on research findings (2020-2025) linking these proxies to sinkhole formation.
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import generic_filter
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_slope(dem: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Calculate slope in radians from DEM.
    
    Args:
        dem: Digital Elevation Model array
        pixel_size: Pixel size in meters
        
    Returns:
        Slope array in radians
    """
    # Sobel gradients for smoother result
    dy = ndimage.sobel(dem, axis=0) / (8 * pixel_size)
    dx = ndimage.sobel(dem, axis=1) / (8 * pixel_size)
    
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    return slope


def calculate_aspect(dem: np.ndarray) -> np.ndarray:
    """
    Calculate aspect (slope direction) in radians.
    
    North = 0, East = π/2, South = π, West = 3π/2
    
    Args:
        dem: Digital Elevation Model array
        
    Returns:
        Aspect array in radians (0 to 2π)
    """
    dy = ndimage.sobel(dem, axis=0)
    dx = ndimage.sobel(dem, axis=1)
    
    aspect = np.arctan2(-dx, dy)  # Note: -dx because east is positive x
    
    # Convert from -π to π to 0 to 2π
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    
    return aspect


def calculate_convergence_index(dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate Convergence Index - measures flow concentration.
    
    Sinkholes form where water collects. High convergence (negative values)
    indicates terrain "pointing" toward a pixel - water pooling zones.
    
    Based on findings from Turkey and Texas sinkhole studies.
    
    Args:
        dem: Digital Elevation Model array
        window_size: Size of analysis window (default 3x3)
        
    Returns:
        Convergence index array (negative = convergent, positive = divergent)
    """
    aspect = calculate_aspect(dem)
    
    rows, cols = dem.shape
    pad = window_size // 2
    
    # Pad arrays
    aspect_padded = np.pad(aspect, pad, mode='reflect')
    
    # For each pixel, calculate how much neighbors "point" toward center
    convergence = np.zeros_like(dem)
    
    for i in range(rows):
        for j in range(cols):
            # Get window of aspects
            window = aspect_padded[i:i+window_size, j:j+window_size]
            
            # Calculate direction from each neighbor to center
            center_row, center_col = pad, pad
            
            total_diff = 0
            count = 0
            
            for wi in range(window_size):
                for wj in range(window_size):
                    if wi == center_row and wj == center_col:
                        continue
                    
                    # Direction from neighbor to center
                    dy = center_row - wi
                    dx = center_col - wj
                    direction_to_center = np.arctan2(dx, -dy)
                    if direction_to_center < 0:
                        direction_to_center += 2 * np.pi
                    
                    # Aspect of neighbor
                    neighbor_aspect = window[wi, wj]
                    
                    # Angular difference (how aligned is flow with direction to center)
                    diff = abs(neighbor_aspect - direction_to_center)
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    
                    # Convert to convergence contribution
                    # 0 diff = fully convergent (-1), π diff = fully divergent (+1)
                    contribution = (diff / np.pi) * 2 - 1
                    total_diff += contribution
                    count += 1
            
            convergence[i, j] = total_diff / max(count, 1) * 100  # Scale to -100 to +100
    
    return convergence


def calculate_convergence_index_fast(dem: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
    """
    Fast approximation of Convergence Index using curvature.
    
    Plan curvature is a good proxy for flow convergence:
    - Negative curvature = convergent (collecting water)
    - Positive curvature = divergent (shedding water)
    
    Args:
        dem: Digital Elevation Model array
        pixel_size: Pixel size in meters
        
    Returns:
        Convergence proxy (negative = convergent = sinkhole likely)
    """
    # Second derivatives
    d2z_dx2 = ndimage.sobel(ndimage.sobel(dem, axis=1), axis=1) / (pixel_size ** 2)
    d2z_dy2 = ndimage.sobel(ndimage.sobel(dem, axis=0), axis=0) / (pixel_size ** 2)
    d2z_dxdy = ndimage.sobel(ndimage.sobel(dem, axis=0), axis=1) / (pixel_size ** 2)
    
    dz_dx = ndimage.sobel(dem, axis=1) / pixel_size
    dz_dy = ndimage.sobel(dem, axis=0) / pixel_size
    
    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = d2z_dxdy
    t = d2z_dy2
    
    # Plan curvature formula
    denom = (p**2 + q**2) * np.sqrt(1 + p**2 + q**2)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    
    plan_curv = -(q**2 * r - 2 * p * q * s + p**2 * t) / denom
    
    return plan_curv * 1000  # Scale for readability


def calculate_topographic_wetness_index(dem: np.ndarray, 
                                        pixel_size: float,
                                        flow_accumulation: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Topographic Wetness Index (TWI).
    
    TWI = ln(a / tan(β))
    
    Where:
    - a = specific catchment area (upslope contributing area per unit contour length)
    - β = local slope
    
    High TWI = wet zone = higher dissolution risk for karst.
    
    Args:
        dem: Digital Elevation Model array
        pixel_size: Pixel size in meters
        flow_accumulation: Optional pre-computed flow accumulation
        
    Returns:
        TWI array (higher = wetter terrain)
    """
    slope = calculate_slope(dem, pixel_size)
    
    # Compute flow accumulation if not provided
    if flow_accumulation is None:
        flow_accumulation = compute_simple_flow_accumulation(dem)
    
    # Specific catchment area (flow acc * pixel area / contour length)
    # For a square pixel, contour length ≈ pixel_size
    specific_catchment = (flow_accumulation + 1) * pixel_size  # +1 to include self
    
    # Avoid division by zero in flat areas
    tan_slope = np.tan(slope)
    tan_slope = np.where(tan_slope < 0.001, 0.001, tan_slope)
    
    twi = np.log(specific_catchment / tan_slope)
    
    return twi


def compute_simple_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """
    Simple D8 flow accumulation algorithm.
    
    For production use, consider using pysheds or richdem for proper handling.
    """
    rows, cols = dem.shape
    flow_acc = np.ones_like(dem)
    
    # D8 directions: E, SE, S, SW, W, NW, N, NE
    di = [0, 1, 1, 1, 0, -1, -1, -1]
    dj = [1, 1, 0, -1, -1, -1, 0, 1]
    
    # Simple iterative approach (not optimal but functional)
    # Sort cells by elevation (high to low)
    flat_indices = np.argsort(dem.ravel())[::-1]
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, dem.shape)
        
        # Find steepest downslope neighbor
        min_elev = dem[i, j]
        min_dir = -1
        
        for d in range(8):
            ni, nj = i + di[d], j + dj[d]
            if 0 <= ni < rows and 0 <= nj < cols:
                if dem[ni, nj] < min_elev:
                    min_elev = dem[ni, nj]
                    min_dir = d
        
        # Add flow to downslope neighbor
        if min_dir >= 0:
            ni, nj = i + di[min_dir], j + dj[min_dir]
            flow_acc[ni, nj] += flow_acc[i, j]
    
    return flow_acc


def calculate_terrain_ruggedness_index(dem: np.ndarray) -> np.ndarray:
    """
    Calculate Terrain Ruggedness Index (TRI).
    
    TRI = sqrt(sum of squared elevation differences to 8 neighbors)
    
    Anomalous roughness in flat terrain may indicate active raveling
    or surface pitting before major collapse.
    
    Args:
        dem: Digital Elevation Model array
        
    Returns:
        TRI array (higher = rougher terrain)
    """
    def tri_kernel(window):
        center = window[4]  # Center of 3x3 window
        diffs_sq = (window - center) ** 2
        # Exclude center from sum
        diffs_sq[4] = 0
        return np.sqrt(np.sum(diffs_sq))
    
    # Apply kernel
    tri = generic_filter(dem, tri_kernel, size=3, mode='reflect')
    
    return tri


def calculate_all_proxies(dem_path: str,
                         output_dir: str) -> Dict[str, str]:
    """
    Calculate all geological proxies from a DEM and save as GeoTIFFs.
    
    Args:
        dem_path: Path to input DEM GeoTIFF
        output_dir: Directory to save output proxies
        
    Returns:
        Dictionary mapping proxy names to output paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        transform = src.transform
        
        # Calculate pixel size in meters
        if src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            pixel_size = transform[0] * 111320 * np.cos(np.deg2rad(center_lat))
        else:
            pixel_size = transform[0]
    
    outputs = {}
    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    
    # 1. Slope
    logger.info("Calculating slope...")
    slope = calculate_slope(dem, pixel_size)
    slope_path = os.path.join(output_dir, 'slope.tif')
    with rasterio.open(slope_path, 'w', **profile) as dst:
        dst.write(np.degrees(slope).astype(np.float32), 1)
        dst.set_band_description(1, "Slope (degrees)")
    outputs['slope'] = slope_path
    
    # 2. Convergence Index (fast version)
    logger.info("Calculating convergence index...")
    ci = calculate_convergence_index_fast(dem, pixel_size)
    ci_path = os.path.join(output_dir, 'convergence_index.tif')
    with rasterio.open(ci_path, 'w', **profile) as dst:
        dst.write(ci.astype(np.float32), 1)
        dst.set_band_description(1, "Convergence Index (negative=convergent)")
    outputs['convergence_index'] = ci_path
    
    # 3. TWI
    logger.info("Calculating topographic wetness index...")
    twi = calculate_topographic_wetness_index(dem, pixel_size)
    twi_path = os.path.join(output_dir, 'twi.tif')
    with rasterio.open(twi_path, 'w', **profile) as dst:
        dst.write(twi.astype(np.float32), 1)
        dst.set_band_description(1, "Topographic Wetness Index")
    outputs['twi'] = twi_path
    
    # 4. TRI
    logger.info("Calculating terrain ruggedness index...")
    tri = calculate_terrain_ruggedness_index(dem)
    tri_path = os.path.join(output_dir, 'tri.tif')
    with rasterio.open(tri_path, 'w', **profile) as dst:
        dst.write(tri.astype(np.float32), 1)
        dst.set_band_description(1, "Terrain Ruggedness Index")
    outputs['tri'] = tri_path
    
    logger.info(f"Saved all proxies to {output_dir}")
    
    return outputs


def extract_proxy_values_at_points(proxy_paths: Dict[str, str],
                                   points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract proxy values at specific point locations.
    
    Args:
        proxy_paths: Dictionary of proxy name -> file path
        points: Nx2 array of (lat, lon) coordinates
        
    Returns:
        Dictionary of proxy name -> values array
    """
    results = {}
    
    for proxy_name, path in proxy_paths.items():
        with rasterio.open(path) as src:
            values = []
            for lat, lon in points:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
                else:
                    val = np.nan
                values.append(val)
            results[proxy_name] = np.array(values)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate geological proxies from DEM")
    parser.add_argument("--dem", required=True, help="Input DEM GeoTIFF")
    parser.add_argument("--output-dir", default="proxies", help="Output directory")
    
    args = parser.parse_args()
    
    import os
    if os.path.exists(args.dem):
        outputs = calculate_all_proxies(args.dem, args.output_dir)
        print(f"\nGenerated proxies:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    else:
        logger.error(f"DEM file '{args.dem}' not found.")
