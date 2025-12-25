"""
GeoRisk Insurance Risk Engine
=============================

Generates insurance-grade risk assessments and data products
for property, portfolio, and parametric applications.

Output Products:
1. Property Risk Reports - Per-asset sinkhole risk scoring
2. Portfolio Analytics - Aggregate exposure analysis
3. Parametric Trigger Data - Satellite-verified thresholds for payouts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Insurance risk tier classification."""
    TIER_1_MINIMAL = "minimal"      # Score 0-20
    TIER_2_LOW = "low"              # Score 21-40
    TIER_3_MODERATE = "moderate"    # Score 41-60
    TIER_4_ELEVATED = "elevated"    # Score 61-80
    TIER_5_SEVERE = "severe"        # Score 81-100


@dataclass
class PropertyLocation:
    """Property location for risk assessment."""
    property_id: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    insured_value: Optional[float] = None
    policy_number: Optional[str] = None
    property_type: Optional[str] = None  # residential, commercial, industrial


@dataclass
class PropertyRiskReport:
    """
    Insurance-ready property risk report.
    
    Contains all metrics needed for:
    - Underwriting decisions
    - Premium calculations
    - Monitoring recommendations
    """
    # Identification
    property_id: str
    latitude: float
    longitude: float
    report_date: str
    
    # Risk Scores (0-100)
    composite_risk_score: float
    velocity_risk_score: float
    density_risk_score: float
    geological_risk_score: float
    proximity_risk_score: float
    
    # Classification
    risk_tier: str
    monitoring_recommendation: str
    
    # Raw Metrics
    velocity_mm_yr: Optional[float] = None
    velocity_trend: Optional[str] = None  # accelerating, stable, decelerating
    density_contrast_kg_m3: Optional[float] = None
    distance_to_nearest_sinkhole_km: Optional[float] = None
    cluster_id: Optional[int] = None
    
    # Geological Context
    geological_formation: Optional[str] = None
    karst_susceptibility: Optional[str] = None
    groundwater_depth_m: Optional[float] = None
    
    # Historical
    historical_sinkholes_5km: int = 0
    last_sinkhole_years: Optional[float] = None
    
    # Insurance Data
    insured_value: Optional[float] = None
    expected_loss_ratio: Optional[float] = None
    recommended_premium_adjustment: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'property_id': self.property_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'report_date': self.report_date,
            'composite_risk_score': self.composite_risk_score,
            'velocity_risk_score': self.velocity_risk_score,
            'density_risk_score': self.density_risk_score,
            'geological_risk_score': self.geological_risk_score,
            'proximity_risk_score': self.proximity_risk_score,
            'risk_tier': self.risk_tier,
            'monitoring_recommendation': self.monitoring_recommendation,
            'velocity_mm_yr': self.velocity_mm_yr,
            'velocity_trend': self.velocity_trend,
            'density_contrast_kg_m3': self.density_contrast_kg_m3,
            'distance_to_nearest_sinkhole_km': self.distance_to_nearest_sinkhole_km,
            'cluster_id': self.cluster_id,
            'geological_formation': self.geological_formation,
            'karst_susceptibility': self.karst_susceptibility,
            'groundwater_depth_m': self.groundwater_depth_m,
            'historical_sinkholes_5km': self.historical_sinkholes_5km,
            'last_sinkhole_years': self.last_sinkhole_years,
            'insured_value': self.insured_value,
            'expected_loss_ratio': self.expected_loss_ratio,
            'recommended_premium_adjustment': self.recommended_premium_adjustment
        }
    
    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass 
class PortfolioAnalytics:
    """
    Portfolio-level risk analytics for reinsurance.
    """
    portfolio_id: str
    analysis_date: str
    total_properties: int
    total_insured_value: float
    
    # Risk Distribution
    tier_1_count: int = 0
    tier_2_count: int = 0
    tier_3_count: int = 0
    tier_4_count: int = 0
    tier_5_count: int = 0
    
    tier_1_value: float = 0.0
    tier_2_value: float = 0.0
    tier_3_value: float = 0.0
    tier_4_value: float = 0.0
    tier_5_value: float = 0.0
    
    # Concentration Risk
    concentration_zones: List[Dict] = field(default_factory=list)
    max_zone_exposure: float = 0.0
    
    # Aggregate Metrics
    weighted_avg_risk_score: float = 0.0
    expected_annual_loss: float = 0.0
    probable_maximum_loss: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ParametricTrigger:
    """
    Parametric trigger definition for automated payouts.
    """
    trigger_id: str
    location_lat: float
    location_lon: float
    radius_km: float
    
    # Trigger Conditions
    velocity_threshold_mm_yr: float
    acceleration_threshold_pct: float
    confirmation_period_days: int
    
    # Payout Structure
    payout_amount: float
    payout_currency: str = "USD"
    
    # Verification
    data_source: str = "Sentinel-1 InSAR"
    oracle_provider: Optional[str] = None
    
    # Status
    is_triggered: bool = False
    trigger_date: Optional[str] = None
    measured_velocity: Optional[float] = None


class RiskEngine:
    """
    Core risk scoring engine for sinkhole hazard assessment.
    
    Combines multiple data inputs to generate composite risk scores:
    - InSAR velocity measurements
    - Gravity-derived density anomalies
    - Geological susceptibility layers
    - Historical sinkhole inventory
    - Spectral anomaly indices
    """
    
    def __init__(self, region: str = 'florida_karst'):
        self.region = region
        self.weights = self._get_region_weights(region)
        logger.info(f"RiskEngine initialized for region: {region}")
    
    def _get_region_weights(self, region: str) -> Dict[str, float]:
        """
        Get region-specific weights for risk components.
        
        Different geological settings prioritize different indicators.
        """
        weights = {
            'florida_karst': {
                'velocity': 0.35,
                'density': 0.25,
                'geological': 0.20,
                'proximity': 0.15,
                'spectral': 0.05
            },
            'texas_salt': {
                'velocity': 0.30,
                'horizontal_strain': 0.25,  # Critical for salt domes
                'density': 0.20,
                'geological': 0.15,
                'proximity': 0.10
            },
            'general': {
                'velocity': 0.30,
                'density': 0.25,
                'geological': 0.25,
                'proximity': 0.15,
                'spectral': 0.05
            }
        }
        return weights.get(region, weights['general'])
    
    def calculate_velocity_risk_score(self, 
                                     velocity_mm_yr: float,
                                     region: Optional[str] = None) -> float:
        """
        Calculate velocity-based risk score (0-100).
        
        Uses region-specific thresholds from literature.
        """
        if region is None:
            region = self.region
        
        # Regional thresholds
        thresholds = {
            'florida_karst': {'warning': -3.0, 'critical': -6.0, 'imminent': -12.0},
            'texas_salt': {'warning': -10.0, 'critical': -20.0, 'imminent': -30.0},
            'general': {'warning': -5.0, 'critical': -10.0, 'imminent': -20.0}
        }.get(region, {'warning': -5.0, 'critical': -10.0, 'imminent': -20.0})
        
        # Noise floor
        if velocity_mm_yr > -2.0:
            return 0.0
        
        # Linear interpolation between thresholds
        if velocity_mm_yr > thresholds['warning']:
            # Between noise and warning: 0-30
            score = 30 * (-velocity_mm_yr - 2) / (-thresholds['warning'] - 2)
        elif velocity_mm_yr > thresholds['critical']:
            # Between warning and critical: 30-60
            score = 30 + 30 * (-velocity_mm_yr - (-thresholds['warning'])) / (
                -thresholds['critical'] - (-thresholds['warning']))
        elif velocity_mm_yr > thresholds['imminent']:
            # Between critical and imminent: 60-85
            score = 60 + 25 * (-velocity_mm_yr - (-thresholds['critical'])) / (
                -thresholds['imminent'] - (-thresholds['critical']))
        else:
            # Beyond imminent: 85-100
            score = 85 + min(15, 15 * (-velocity_mm_yr - (-thresholds['imminent'])) / 10)
        
        return min(100, max(0, score))
    
    def calculate_density_risk_score(self, 
                                    density_contrast_kg_m3: float) -> float:
        """
        Calculate density-based risk score (0-100).
        
        More negative density = more void = higher risk.
        """
        # Limestone void threshold ~-2670 kg/m³ (full void)
        # Realistic cavities rarely show full contrast
        
        if density_contrast_kg_m3 > -20:
            return 0.0  # No significant void
        elif density_contrast_kg_m3 > -50:
            return 20 * (-density_contrast_kg_m3 - 20) / 30
        elif density_contrast_kg_m3 > -100:
            return 20 + 30 * (-density_contrast_kg_m3 - 50) / 50
        elif density_contrast_kg_m3 > -200:
            return 50 + 30 * (-density_contrast_kg_m3 - 100) / 100
        else:
            return 80 + min(20, 20 * (-density_contrast_kg_m3 - 200) / 200)
    
    def calculate_proximity_risk_score(self,
                                       distance_to_sinkhole_km: float,
                                       historical_count_5km: int = 0) -> float:
        """
        Calculate proximity-based risk score (0-100).
        
        Based on Konya Basin research: sinkholes cluster within ~500m
        and new events occur within 11 months of neighbors.
        """
        score = 0.0
        
        # Distance component
        if distance_to_sinkhole_km < 0.5:  # Within 500m
            score += 50
        elif distance_to_sinkhole_km < 1.0:
            score += 35
        elif distance_to_sinkhole_km < 2.0:
            score += 20
        elif distance_to_sinkhole_km < 5.0:
            score += 10
        
        # Historical density component
        if historical_count_5km > 10:
            score += 30
        elif historical_count_5km > 5:
            score += 20
        elif historical_count_5km > 2:
            score += 10
        elif historical_count_5km > 0:
            score += 5
        
        return min(100, score)
    
    def calculate_geological_risk_score(self,
                                        karst_susceptibility: str = 'unknown',
                                        lithology: str = 'unknown',
                                        groundwater_depth_m: Optional[float] = None) -> float:
        """
        Calculate geological context risk score (0-100).
        """
        score = 0.0
        
        # Karst susceptibility
        susceptibility_scores = {
            'very_high': 40,
            'high': 30,
            'moderate': 20,
            'low': 10,
            'none': 0,
            'unknown': 15
        }
        score += susceptibility_scores.get(karst_susceptibility.lower(), 15)
        
        # Lithology
        lithology_scores = {
            'limestone': 30,
            'dolomite': 28,
            'gypsum': 35,
            'salt': 35,
            'sandstone': 10,
            'shale': 5,
            'granite': 0,
            'unknown': 15
        }
        score += lithology_scores.get(lithology.lower(), 15)
        
        # Groundwater depth (shallower = higher dissolution risk)
        if groundwater_depth_m is not None:
            if groundwater_depth_m < 10:
                score += 20
            elif groundwater_depth_m < 30:
                score += 15
            elif groundwater_depth_m < 60:
                score += 10
            else:
                score += 5
        
        return min(100, score)
    
    def generate_property_report(self,
                                 property: PropertyLocation,
                                 velocity_mm_yr: Optional[float] = None,
                                 density_contrast: Optional[float] = None,
                                 distance_to_sinkhole: Optional[float] = None,
                                 historical_count: int = 0,
                                 geological_data: Optional[Dict] = None,
                                 cluster_id: Optional[int] = None) -> PropertyRiskReport:
        """
        Generate a comprehensive property risk report.
        """
        # Calculate component scores
        velocity_score = self.calculate_velocity_risk_score(velocity_mm_yr) if velocity_mm_yr else 0
        density_score = self.calculate_density_risk_score(density_contrast) if density_contrast else 0
        proximity_score = self.calculate_proximity_risk_score(
            distance_to_sinkhole if distance_to_sinkhole else 100,
            historical_count
        )
        
        geological_data = geological_data or {}
        geological_score = self.calculate_geological_risk_score(
            geological_data.get('karst_susceptibility', 'unknown'),
            geological_data.get('lithology', 'unknown'),
            geological_data.get('groundwater_depth_m')
        )
        
        # Calculate composite score
        weights = self.weights
        composite_score = (
            weights.get('velocity', 0.30) * velocity_score +
            weights.get('density', 0.25) * density_score +
            weights.get('proximity', 0.15) * proximity_score +
            weights.get('geological', 0.25) * geological_score
        )
        
        # Determine risk tier
        if composite_score <= 20:
            risk_tier = RiskTier.TIER_1_MINIMAL.value
            monitoring = "Standard monitoring - annual review"
        elif composite_score <= 40:
            risk_tier = RiskTier.TIER_2_LOW.value
            monitoring = "Enhanced monitoring - semi-annual review"
        elif composite_score <= 60:
            risk_tier = RiskTier.TIER_3_MODERATE.value
            monitoring = "Active monitoring - quarterly InSAR review"
        elif composite_score <= 80:
            risk_tier = RiskTier.TIER_4_ELEVATED.value
            monitoring = "High-frequency monitoring - monthly InSAR + ground inspection"
        else:
            risk_tier = RiskTier.TIER_5_SEVERE.value
            monitoring = "CRITICAL - Immediate ground-truth investigation required"
        
        # Calculate expected loss metrics if insured value provided
        expected_loss_ratio = None
        premium_adjustment = None
        if property.insured_value:
            # Simplified model: risk score maps to loss probability
            loss_probability = composite_score / 1000  # 0-10% base probability
            expected_loss_ratio = loss_probability
            premium_adjustment = 1.0 + (composite_score / 50)  # 1.0x to 3.0x
        
        report = PropertyRiskReport(
            property_id=property.property_id,
            latitude=property.latitude,
            longitude=property.longitude,
            report_date=datetime.now().isoformat(),
            composite_risk_score=round(composite_score, 1),
            velocity_risk_score=round(velocity_score, 1),
            density_risk_score=round(density_score, 1),
            geological_risk_score=round(geological_score, 1),
            proximity_risk_score=round(proximity_score, 1),
            risk_tier=risk_tier,
            monitoring_recommendation=monitoring,
            velocity_mm_yr=velocity_mm_yr,
            density_contrast_kg_m3=density_contrast,
            distance_to_nearest_sinkhole_km=distance_to_sinkhole,
            cluster_id=cluster_id,
            geological_formation=geological_data.get('lithology'),
            karst_susceptibility=geological_data.get('karst_susceptibility'),
            groundwater_depth_m=geological_data.get('groundwater_depth_m'),
            historical_sinkholes_5km=historical_count,
            insured_value=property.insured_value,
            expected_loss_ratio=expected_loss_ratio,
            recommended_premium_adjustment=premium_adjustment
        )
        
        return report
    
    def generate_portfolio_analytics(self,
                                     reports: List[PropertyRiskReport],
                                     portfolio_id: str) -> PortfolioAnalytics:
        """
        Generate portfolio-level analytics from property reports.
        """
        analytics = PortfolioAnalytics(
            portfolio_id=portfolio_id,
            analysis_date=datetime.now().isoformat(),
            total_properties=len(reports),
            total_insured_value=sum(r.insured_value or 0 for r in reports)
        )
        
        # Count by tier
        for report in reports:
            value = report.insured_value or 0
            if report.risk_tier == RiskTier.TIER_1_MINIMAL.value:
                analytics.tier_1_count += 1
                analytics.tier_1_value += value
            elif report.risk_tier == RiskTier.TIER_2_LOW.value:
                analytics.tier_2_count += 1
                analytics.tier_2_value += value
            elif report.risk_tier == RiskTier.TIER_3_MODERATE.value:
                analytics.tier_3_count += 1
                analytics.tier_3_value += value
            elif report.risk_tier == RiskTier.TIER_4_ELEVATED.value:
                analytics.tier_4_count += 1
                analytics.tier_4_value += value
            elif report.risk_tier == RiskTier.TIER_5_SEVERE.value:
                analytics.tier_5_count += 1
                analytics.tier_5_value += value
        
        # Weighted average risk score
        if analytics.total_insured_value > 0:
            weighted_sum = sum(
                r.composite_risk_score * (r.insured_value or 0) 
                for r in reports
            )
            analytics.weighted_avg_risk_score = weighted_sum / analytics.total_insured_value
        
        # Expected annual loss (simplified)
        analytics.expected_annual_loss = sum(
            (r.expected_loss_ratio or 0) * (r.insured_value or 0)
            for r in reports
        )
        
        # Probable Maximum Loss (top 10% of risk)
        sorted_reports = sorted(reports, key=lambda x: x.composite_risk_score, reverse=True)
        top_10_pct = sorted_reports[:max(1, len(reports) // 10)]
        analytics.probable_maximum_loss = sum(r.insured_value or 0 for r in top_10_pct)
        
        return analytics
    
    def create_parametric_trigger(self,
                                  location_lat: float,
                                  location_lon: float,
                                  payout_amount: float,
                                  velocity_threshold: Optional[float] = None,
                                  radius_km: float = 1.0) -> ParametricTrigger:
        """
        Create a parametric trigger definition for a location.
        
        Uses region-specific thresholds if not specified.
        """
        if velocity_threshold is None:
            # Default to critical threshold for region
            thresholds = {
                'florida_karst': -6.0,
                'texas_salt': -20.0,
                'general': -10.0
            }
            velocity_threshold = thresholds.get(self.region, -10.0)
        
        trigger = ParametricTrigger(
            trigger_id=f"PARAM_{location_lat:.4f}_{location_lon:.4f}_{datetime.now().strftime('%Y%m%d')}",
            location_lat=location_lat,
            location_lon=location_lon,
            radius_km=radius_km,
            velocity_threshold_mm_yr=velocity_threshold,
            acceleration_threshold_pct=50.0,  # 50% acceleration = acute
            confirmation_period_days=30,
            payout_amount=payout_amount,
            data_source="Sentinel-1 InSAR",
        )
        
        return trigger


# Export functions for batch processing
def process_property_batch(properties: List[Dict],
                          velocity_data: Dict[str, float],
                          density_data: Dict[str, float],
                          region: str = 'florida_karst') -> pd.DataFrame:
    """
    Process a batch of properties and return DataFrame of reports.
    """
    engine = RiskEngine(region=region)
    reports = []
    
    for prop_data in properties:
        prop = PropertyLocation(
            property_id=prop_data['id'],
            latitude=prop_data['lat'],
            longitude=prop_data['lon'],
            insured_value=prop_data.get('value')
        )
        
        report = engine.generate_property_report(
            property=prop,
            velocity_mm_yr=velocity_data.get(prop_data['id']),
            density_contrast=density_data.get(prop_data['id'])
        )
        
        reports.append(report.to_dict())
    
    return pd.DataFrame(reports)


if __name__ == "__main__":
    # Demo usage
    engine = RiskEngine(region='florida_karst')
    
    # Example property
    prop = PropertyLocation(
        property_id="FL-001",
        latitude=28.0394,
        longitude=-81.9498,
        insured_value=250000
    )
    
    report = engine.generate_property_report(
        property=prop,
        velocity_mm_yr=-4.5,
        density_contrast=-75.0,
        distance_to_sinkhole=0.8,
        historical_count=3,
        geological_data={
            'karst_susceptibility': 'high',
            'lithology': 'limestone',
            'groundwater_depth_m': 25
        }
    )
    
    print("=== Property Risk Report ===")
    print(report.to_json())
    
    print(f"\nComposite Risk Score: {report.composite_risk_score}")
    print(f"Risk Tier: {report.risk_tier}")
    print(f"Monitoring: {report.monitoring_recommendation}")
