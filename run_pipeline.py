#!/usr/bin/env python3
"""
GeoRisk Pipeline Runner
=======================

Main execution script for the GeoRisk sinkhole detection pipeline.

Modes:
1. Full Pipeline: Gravity + InSAR + Clustering + Risk Scoring
2. Property Assessment: Score specific property locations
3. Portfolio Batch: Process entire property portfolio
4. Parametric Setup: Create trigger definitions

Usage:
    python run_pipeline.py --mode full --region florida_karst --input gravity.tif
    python run_pipeline.py --mode property --input properties.csv --region florida_karst
    python run_pipeline.py --mode portfolio --input portfolio.csv --output report.json
"""

import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# GeoRisk imports
from georisk.core.pinn_sinkhole_inversion import invert_gravity_for_voids, SinkholeInversionConfig
from georisk.core.insar_processor import InSARProcessor
from georisk.core.clustering import SinkholeClusterDetector
from georisk.core.risk_engine import (
    RiskEngine, PropertyLocation, PropertyRiskReport, 
    PortfolioAnalytics, process_property_batch
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeoRisk')


class GeoRiskPipeline:
    """
    Main GeoRisk pipeline orchestrator.
    
    Coordinates data processing from satellite inputs through
    to insurance-ready risk reports.
    """
    
    def __init__(self, 
                 region: str = 'florida_karst',
                 output_dir: str = 'georisk/data/outputs',
                 config_path: Optional[str] = None):
        """
        Initialize the GeoRisk pipeline.
        
        Args:
            region: Geological region for threshold calibration
            output_dir: Directory for output files
            config_path: Optional path to configuration file
        """
        self.region = region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.risk_engine = RiskEngine(region=region)
        self.insar_processor = InSARProcessor(region=region)
        self.cluster_detector = SinkholeClusterDetector()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        logger.info(f"GeoRisk Pipeline initialized for region: {region}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        default_config = {
            'gravity_inversion': {
                'epochs': 500,
                'depth_estimate': 50.0,
                'thickness_estimate': 30.0
            },
            'clustering': {
                'min_cluster_size': 5,
                'eps_meters': 30.0,
                'velocity_threshold': -3.0
            },
            'reporting': {
                'include_coordinates': True,
                'include_raw_metrics': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                default_config.update(loaded)
        
        return default_config
    
    def run_full_pipeline(self,
                         gravity_path: Optional[str] = None,
                         velocity_path: Optional[str] = None,
                         dem_path: Optional[str] = None) -> dict:
        """
        Run the complete GeoRisk processing pipeline.
        
        Args:
            gravity_path: Path to Bouguer gravity GeoTIFF
            velocity_path: Path to InSAR velocity GeoTIFF
            dem_path: Optional DEM for geological proxies
            
        Returns:
            Dictionary containing all outputs and metadata
        """
        results = {
            'pipeline_start': datetime.now().isoformat(),
            'region': self.region,
            'inputs': {},
            'outputs': {}
        }
        
        # Step 1: Gravity Inversion (if gravity data provided)
        density_path = None
        if gravity_path and os.path.exists(gravity_path):
            logger.info("Step 1: Running gravity inversion for void detection...")
            results['inputs']['gravity'] = gravity_path
            
            density_path = str(self.output_dir / 'void_density_map.tif')
            
            config = SinkholeInversionConfig()
            config.EPOCHS = self.config['gravity_inversion']['epochs']
            config.DEPTH_ESTIMATE = self.config['gravity_inversion']['depth_estimate']
            
            inversion_result = invert_gravity_for_voids(
                gravity_path, 
                density_path,
                config=config,
                region=self.region
            )
            
            results['outputs']['density_map'] = density_path
            results['inversion_metadata'] = inversion_result
        else:
            logger.info("Step 1: Skipping gravity inversion (no data provided)")
        
        # Step 2: InSAR Velocity Analysis (if velocity data provided)
        risk_map_path = None
        if velocity_path and os.path.exists(velocity_path):
            logger.info("Step 2: Processing InSAR velocity data...")
            results['inputs']['velocity'] = velocity_path
            
            risk_map_path = str(self.output_dir / 'risk_classification.tif')
            
            velocity_stats = self.insar_processor.process_velocity_raster(
                velocity_path,
                output_risk_path=risk_map_path
            )
            
            results['outputs']['risk_map'] = risk_map_path
            results['velocity_stats'] = velocity_stats
            
            # Extract warning zones
            zones_df = self.insar_processor.extract_warning_zones(velocity_path)
            if not zones_df.empty:
                zones_path = str(self.output_dir / 'warning_zones.csv')
                zones_df.to_csv(zones_path, index=False)
                results['outputs']['warning_zones'] = zones_path
                results['warning_zone_count'] = len(zones_df)
        else:
            logger.info("Step 2: Skipping InSAR analysis (no data provided)")
        
        # Step 3: Cluster Detection
        if velocity_path and os.path.exists(velocity_path):
            logger.info("Step 3: Running DBSCAN cluster detection...")
            
            clusters = self.cluster_detector.detect_clusters_from_raster(
                velocity_path,
                velocity_threshold=self.config['clustering']['velocity_threshold'],
                density_path=density_path
            )
            
            if clusters:
                clusters_df = self.cluster_detector.clusters_to_dataframe(clusters)
                clusters_path = str(self.output_dir / 'sinkhole_clusters.csv')
                clusters_df.to_csv(clusters_path, index=False)
                results['outputs']['clusters'] = clusters_path
                
                valid_clusters = [c for c in clusters if c.is_valid]
                results['cluster_stats'] = {
                    'total_clusters': len(clusters),
                    'valid_clusters': len(valid_clusters),
                    'max_risk_score': max(c.risk_score for c in clusters) if clusters else 0
                }
                
                # Also export as GeoJSON
                gdf = self.cluster_detector.clusters_to_geodataframe(clusters)
                geojson_path = str(self.output_dir / 'sinkhole_clusters.geojson')
                gdf.to_file(geojson_path, driver='GeoJSON')
                results['outputs']['clusters_geojson'] = geojson_path
        else:
            logger.info("Step 3: Skipping cluster detection (no velocity data)")
        
        # Step 4: Generate Summary Report
        logger.info("Step 4: Generating summary report...")
        
        results['pipeline_end'] = datetime.now().isoformat()
        
        # Save full results as JSON
        results_path = str(self.output_dir / 'pipeline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        results['outputs']['results_json'] = results_path
        
        logger.info(f"Pipeline complete. Results saved to {results_path}")
        
        return results
    
    def assess_properties(self,
                         properties_csv: str,
                         velocity_path: Optional[str] = None,
                         density_path: Optional[str] = None) -> pd.DataFrame:
        """
        Assess risk for a list of properties.
        
        Args:
            properties_csv: CSV with property_id, latitude, longitude, (optional) value
            velocity_path: Optional velocity raster for extraction
            density_path: Optional density raster for extraction
            
        Returns:
            DataFrame of PropertyRiskReports
        """
        logger.info(f"Assessing properties from {properties_csv}")
        
        properties_df = pd.read_csv(properties_csv)
        
        # Standardize column names
        column_mapping = {
            'lat': 'latitude',
            'lon': 'longitude',
            'lng': 'longitude',
            'id': 'property_id',
            'insured_value': 'value'
        }
        properties_df.rename(columns={k: v for k, v in column_mapping.items() 
                                       if k in properties_df.columns}, inplace=True)
        
        # Extract values from rasters if provided
        velocity_values = {}
        density_values = {}
        
        if velocity_path and os.path.exists(velocity_path):
            import rasterio
            with rasterio.open(velocity_path) as src:
                for idx, row in properties_df.iterrows():
                    try:
                        r, c = src.index(row['longitude'], row['latitude'])
                        if 0 <= r < src.height and 0 <= c < src.width:
                            val = src.read(1)[r, c]
                            velocity_values[row['property_id']] = float(val)
                    except:
                        pass
        
        if density_path and os.path.exists(density_path):
            import rasterio
            with rasterio.open(density_path) as src:
                for idx, row in properties_df.iterrows():
                    try:
                        r, c = src.index(row['longitude'], row['latitude'])
                        if 0 <= r < src.height and 0 <= c < src.width:
                            val = src.read(1)[r, c]
                            density_values[row['property_id']] = float(val)
                    except:
                        pass
        
        # Generate reports
        reports = []
        for idx, row in properties_df.iterrows():
            prop = PropertyLocation(
                property_id=str(row['property_id']),
                latitude=row['latitude'],
                longitude=row['longitude'],
                insured_value=row.get('value')
            )
            
            report = self.risk_engine.generate_property_report(
                property=prop,
                velocity_mm_yr=velocity_values.get(row['property_id']),
                density_contrast=density_values.get(row['property_id']),
                geological_data={'karst_susceptibility': 'moderate'}  # Default
            )
            
            reports.append(report.to_dict())
        
        reports_df = pd.DataFrame(reports)
        
        # Save to CSV
        output_path = str(self.output_dir / 'property_risk_reports.csv')
        reports_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(reports_df)} property reports to {output_path}")
        
        return reports_df
    
    def generate_portfolio_report(self,
                                  reports_df: pd.DataFrame,
                                  portfolio_id: str = 'default') -> PortfolioAnalytics:
        """
        Generate portfolio-level analytics from property reports.
        """
        # Convert DataFrame back to PropertyRiskReport objects
        reports = []
        for _, row in reports_df.iterrows():
            report = PropertyRiskReport(
                property_id=row['property_id'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                report_date=row['report_date'],
                composite_risk_score=row['composite_risk_score'],
                velocity_risk_score=row['velocity_risk_score'],
                density_risk_score=row['density_risk_score'],
                geological_risk_score=row['geological_risk_score'],
                proximity_risk_score=row['proximity_risk_score'],
                risk_tier=row['risk_tier'],
                monitoring_recommendation=row['monitoring_recommendation'],
                insured_value=row.get('insured_value')
            )
            reports.append(report)
        
        analytics = self.risk_engine.generate_portfolio_analytics(reports, portfolio_id)
        
        # Save analytics
        analytics_path = str(self.output_dir / 'portfolio_analytics.json')
        with open(analytics_path, 'w') as f:
            json.dump(analytics.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved portfolio analytics to {analytics_path}")
        
        return analytics


def main():
    parser = argparse.ArgumentParser(
        description="GeoRisk Sinkhole Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline with gravity and velocity data:
    python run_pipeline.py --mode full --gravity gravity.tif --velocity velocity.tif --region florida_karst

  Property assessment from CSV:
    python run_pipeline.py --mode property --properties properties.csv --velocity velocity.tif

  Portfolio batch processing:
    python run_pipeline.py --mode portfolio --properties portfolio.csv --output-dir ./reports
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['full', 'property', 'portfolio', 'demo'],
                       help='Pipeline execution mode')
    
    parser.add_argument('--region', default='florida_karst',
                       choices=['florida_karst', 'texas_salt', 'dead_sea', 'konya_basin', 'general'],
                       help='Geological region for threshold calibration')
    
    parser.add_argument('--gravity', help='Bouguer gravity GeoTIFF path')
    parser.add_argument('--velocity', help='InSAR velocity GeoTIFF path')
    parser.add_argument('--dem', help='Digital Elevation Model GeoTIFF path')
    parser.add_argument('--properties', help='Properties CSV path')
    
    parser.add_argument('--output-dir', default='georisk/data/outputs',
                       help='Output directory for results')
    parser.add_argument('--config', help='Configuration YAML file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GeoRiskPipeline(
        region=args.region,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    if args.mode == 'full':
        results = pipeline.run_full_pipeline(
            gravity_path=args.gravity,
            velocity_path=args.velocity,
            dem_path=args.dem
        )
        print("\n=== Pipeline Results ===")
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == 'property':
        if not args.properties:
            parser.error("--properties required for property mode")
        
        reports_df = pipeline.assess_properties(
            properties_csv=args.properties,
            velocity_path=args.velocity
        )
        print(f"\n=== Processed {len(reports_df)} Properties ===")
        print(reports_df[['property_id', 'composite_risk_score', 'risk_tier']].to_string())
        
    elif args.mode == 'portfolio':
        if not args.properties:
            parser.error("--properties required for portfolio mode")
        
        reports_df = pipeline.assess_properties(
            properties_csv=args.properties,
            velocity_path=args.velocity
        )
        analytics = pipeline.generate_portfolio_report(reports_df)
        print("\n=== Portfolio Analytics ===")
        print(json.dumps(analytics.to_dict(), indent=2, default=str))
        
    elif args.mode == 'demo':
        # Run demo with synthetic data
        logger.info("Running demo mode with synthetic examples...")
        
        # Demo property assessment
        demo_props = [
            PropertyLocation("DEMO-001", 28.0394, -81.9498, insured_value=250000),
            PropertyLocation("DEMO-002", 28.0456, -81.9512, insured_value=350000),
            PropertyLocation("DEMO-003", 28.0321, -81.9434, insured_value=175000),
        ]
        
        demo_velocities = {"DEMO-001": -4.5, "DEMO-002": -2.1, "DEMO-003": -8.2}
        demo_densities = {"DEMO-001": -75.0, "DEMO-002": -25.0, "DEMO-003": -120.0}
        
        print("\n=== Demo Property Risk Reports ===")
        for prop in demo_props:
            report = pipeline.risk_engine.generate_property_report(
                property=prop,
                velocity_mm_yr=demo_velocities.get(prop.property_id),
                density_contrast=demo_densities.get(prop.property_id),
                geological_data={
                    'karst_susceptibility': 'high',
                    'lithology': 'limestone',
                    'groundwater_depth_m': 25
                }
            )
            print(f"\nProperty: {report.property_id}")
            print(f"  Risk Score: {report.composite_risk_score}")
            print(f"  Risk Tier: {report.risk_tier}")
            print(f"  Velocity: {report.velocity_mm_yr} mm/yr")
            print(f"  Monitoring: {report.monitoring_recommendation}")


if __name__ == "__main__":
    main()
