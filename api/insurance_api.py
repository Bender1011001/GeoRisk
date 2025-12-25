"""
GeoRisk Insurance API
=====================

REST API for insurance company integration.

Endpoints:
- POST /assess/property - Single property risk assessment
- POST /assess/batch - Batch property assessment
- GET /portfolio/{id}/analytics - Portfolio-level analytics
- POST /triggers/create - Create parametric trigger
- GET /triggers/{id}/status - Check trigger status
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging
import json

# GeoRisk imports
import sys
sys.path.insert(0, '..')
from georisk.core.risk_engine import (
    RiskEngine, PropertyLocation, PropertyRiskReport, 
    ParametricTrigger, RiskTier
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GeoRisk API",
    description="Sinkhole Risk Assessment API for Insurance Companies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize risk engine (default region)
risk_engine = RiskEngine(region='florida_karst')


# ==========================================
# Pydantic Models
# ==========================================

class PropertyInput(BaseModel):
    """Input model for property assessment."""
    property_id: str = Field(..., description="Unique property identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude")
    insured_value: Optional[float] = Field(None, ge=0, description="Insured value in USD")
    address: Optional[str] = None
    property_type: Optional[str] = None

class AssessmentInput(BaseModel):
    """Full assessment input with optional data."""
    property: PropertyInput
    velocity_mm_yr: Optional[float] = Field(None, description="InSAR velocity (mm/yr)")
    density_contrast: Optional[float] = Field(None, description="Density contrast (kg/m³)")
    distance_to_sinkhole_km: Optional[float] = Field(None, description="Distance to nearest sinkhole")
    historical_count: int = Field(0, ge=0, description="Historical sinkholes within 5km")
    region: str = Field("florida_karst", description="Geological region")
    geological_data: Optional[Dict] = None

class BatchAssessmentInput(BaseModel):
    """Batch assessment input."""
    properties: List[AssessmentInput]
    region: str = Field("florida_karst", description="Geological region")

class TriggerCreateInput(BaseModel):
    """Parametric trigger creation input."""
    location_lat: float
    location_lon: float
    radius_km: float = Field(1.0, ge=0.1, le=10)
    velocity_threshold_mm_yr: Optional[float] = None
    payout_amount: float = Field(..., ge=0)
    payout_currency: str = Field("USD")

class RiskReportOutput(BaseModel):
    """Risk report output model."""
    property_id: str
    latitude: float
    longitude: float
    report_date: str
    composite_risk_score: float
    velocity_risk_score: float
    density_risk_score: float
    geological_risk_score: float
    proximity_risk_score: float
    risk_tier: str
    monitoring_recommendation: str
    velocity_mm_yr: Optional[float]
    density_contrast_kg_m3: Optional[float]
    expected_loss_ratio: Optional[float]
    recommended_premium_adjustment: Optional[float]


# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "GeoRisk API",
        "version": "1.0.0",
        "description": "Sinkhole Risk Assessment for Insurance",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/assess/property", response_model=RiskReportOutput)
async def assess_property(input_data: AssessmentInput):
    """
    Assess sinkhole risk for a single property.
    
    Returns a comprehensive risk report including:
    - Composite risk score (0-100)
    - Component scores (velocity, density, geological, proximity)
    - Risk tier classification
    - Monitoring recommendation
    - Loss ratio and premium adjustment suggestions
    """
    try:
        # Set region-specific engine
        engine = RiskEngine(region=input_data.region)
        
        # Create property location
        prop = PropertyLocation(
            property_id=input_data.property.property_id,
            latitude=input_data.property.latitude,
            longitude=input_data.property.longitude,
            insured_value=input_data.property.insured_value,
            address=input_data.property.address,
            property_type=input_data.property.property_type
        )
        
        # Generate report
        report = engine.generate_property_report(
            property=prop,
            velocity_mm_yr=input_data.velocity_mm_yr,
            density_contrast=input_data.density_contrast,
            distance_to_sinkhole=input_data.distance_to_sinkhole_km,
            historical_count=input_data.historical_count,
            geological_data=input_data.geological_data or {}
        )
        
        return RiskReportOutput(
            property_id=report.property_id,
            latitude=report.latitude,
            longitude=report.longitude,
            report_date=report.report_date,
            composite_risk_score=report.composite_risk_score,
            velocity_risk_score=report.velocity_risk_score,
            density_risk_score=report.density_risk_score,
            geological_risk_score=report.geological_risk_score,
            proximity_risk_score=report.proximity_risk_score,
            risk_tier=report.risk_tier,
            monitoring_recommendation=report.monitoring_recommendation,
            velocity_mm_yr=report.velocity_mm_yr,
            density_contrast_kg_m3=report.density_contrast_kg_m3,
            expected_loss_ratio=report.expected_loss_ratio,
            recommended_premium_adjustment=report.recommended_premium_adjustment
        )
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/batch")
async def assess_batch(input_data: BatchAssessmentInput):
    """
    Batch assess multiple properties.
    
    Returns list of risk reports plus aggregate statistics.
    """
    try:
        engine = RiskEngine(region=input_data.region)
        
        reports = []
        for item in input_data.properties:
            prop = PropertyLocation(
                property_id=item.property.property_id,
                latitude=item.property.latitude,
                longitude=item.property.longitude,
                insured_value=item.property.insured_value
            )
            
            report = engine.generate_property_report(
                property=prop,
                velocity_mm_yr=item.velocity_mm_yr,
                density_contrast=item.density_contrast,
                distance_to_sinkhole=item.distance_to_sinkhole_km,
                historical_count=item.historical_count,
                geological_data=item.geological_data or {}
            )
            
            reports.append(report.to_dict())
        
        # Calculate aggregate statistics
        scores = [r['composite_risk_score'] for r in reports]
        tier_counts = {}
        for r in reports:
            tier = r['risk_tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        return {
            "total_properties": len(reports),
            "reports": reports,
            "aggregate_stats": {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "tier_distribution": tier_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Batch assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triggers/create")
async def create_trigger(input_data: TriggerCreateInput):
    """
    Create a parametric trigger definition.
    
    Triggers automatically when satellite-verified deformation
    exceeds the specified threshold.
    """
    try:
        trigger = risk_engine.create_parametric_trigger(
            location_lat=input_data.location_lat,
            location_lon=input_data.location_lon,
            payout_amount=input_data.payout_amount,
            velocity_threshold=input_data.velocity_threshold_mm_yr,
            radius_km=input_data.radius_km
        )
        
        return {
            "trigger_id": trigger.trigger_id,
            "location": {
                "latitude": trigger.location_lat,
                "longitude": trigger.location_lon,
                "radius_km": trigger.radius_km
            },
            "conditions": {
                "velocity_threshold_mm_yr": trigger.velocity_threshold_mm_yr,
                "acceleration_threshold_pct": trigger.acceleration_threshold_pct,
                "confirmation_period_days": trigger.confirmation_period_days
            },
            "payout": {
                "amount": trigger.payout_amount,
                "currency": trigger.payout_currency
            },
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trigger creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regions")
async def list_regions():
    """List available geological regions with their thresholds."""
    from georisk.core.insar_processor import REGIONAL_THRESHOLDS
    
    regions = {}
    for name, thresholds in REGIONAL_THRESHOLDS.items():
        regions[name] = {
            "description": thresholds.description,
            "sensor_type": thresholds.sensor_type,
            "thresholds": {
                "warning": thresholds.warning_threshold,
                "critical": thresholds.critical_threshold,
                "imminent": thresholds.imminent_threshold
            }
        }
    
    return {"regions": regions}


# ==========================================
# Startup
# ==========================================

@app.on_event("startup")
async def startup_event():
    logger.info("GeoRisk API starting up...")
    logger.info("Available endpoints: /docs, /assess/property, /assess/batch, /triggers/create")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("GeoRisk API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
