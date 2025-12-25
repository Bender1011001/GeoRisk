"""
GeoRisk - AI-Powered Sinkhole Detection for Insurance
"""

__version__ = "1.0.0"
__author__ = "GeoRisk Team"
__description__ = "Physics-Informed Sinkhole Detection & Risk Assessment for Insurance"

from .core.pinn_sinkhole_inversion import (
    SinkholePINN,
    VoidGravityPhysicsLayer,
    invert_gravity_for_voids
)
from .core.insar_processor import InSARProcessor, VelocityAnalyzer
from .core.clustering import SinkholeClusterDetector
from .core.risk_engine import (
    RiskEngine,
    PropertyLocation,
    PropertyRiskReport,
    PortfolioAnalytics,
    ParametricTrigger
)

__all__ = [
    # Core Classes
    'SinkholePINN',
    'VoidGravityPhysicsLayer',
    'InSARProcessor',
    'VelocityAnalyzer',
    'SinkholeClusterDetector',
    'RiskEngine',
    
    # Data Classes
    'PropertyLocation',
    'PropertyRiskReport',
    'PortfolioAnalytics',
    'ParametricTrigger',
    
    # Functions
    'invert_gravity_for_voids',
]
