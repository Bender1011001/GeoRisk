# GeoRisk Scientific Methodology

## Executive Summary

GeoRisk implements a **deterministic** approach to sinkhole hazard assessment, transitioning from traditional probabilistic susceptibility mapping to real-time, satellite-based early warning. This methodology is grounded in peer-reviewed research published between 2020-2025.

---

## 1. The Paradigm Shift

### From Probabilistic to Deterministic

| Probabilistic Approach | Deterministic Approach (GeoRisk) |
|------------------------|----------------------------------|
| "1% annual probability of collapse" | "Velocity at -4.5 mm/yr, accelerating" |
| Static zone assignment | Real-time monitoring |
| Where sinkholes might occur | **When** collapse is imminent |
| Post-event indemnification | Pre-collapse intervention |

### Financial Value for Insurers

- **Acute Risk Identification**: Differentiate chronic subsidence from active void propagation
- **Parametric Triggers**: Objective, satellite-verified payout conditions
- **Loss Mitigation**: Early warning enables ground stabilization before catastrophic failure

---

## 2. The "Golden Window" Concept

Research confirms a detectable precursor period exists before sinkhole collapse. The duration varies by geological setting:

| Region | Precursor Window | Cause |
|--------|-----------------|-------|
| Dead Sea | Days to 5 Years | Viscoelastic creep in cemented alluvium |
| Florida Karst | 1 Month to 1.7 Years | Cover-collapse in sand/limestone |
| Konya Basin | Months (clustered) | Anthropogenic groundwater depletion |
| Salt Domes | Months to Years | Cavity pressure loss, radial compression |

**Key Insight**: The "Golden Window" is the operational timeframe for intervention.

---

## 3. Velocity Thresholds

### Regional Calibration

GeoRisk applies region-specific velocity thresholds based on literature consensus:

#### Florida Karst (Cover-Collapse)
```
Noise Floor:    > -2 mm/yr    → Stable (no action)
Warning Zone:   -3 to -6 mm/yr → Yellow Flag (investigation)
Critical Zone:  < -6 mm/yr     → Red Flag (high probability)
```
**Sensor**: X-band InSAR (TerraSAR-X) for high resolution
**Reference**: Oliver-Cabrera et al. 2022

#### Salt/Evaporite Regions (Texas, Dead Sea)
```
Noise Floor:    > -5 mm/yr     → Stable
Warning Zone:   -10 to -20 mm/yr → Yellow Flag
Critical Zone:  < -20 mm/yr    → Red Flag
```
**Sensor**: C-band (Sentinel-1) or L-band for large deformations
**Reference**: Wink Sink studies, Bayou Corne analysis

---

## 4. Multi-Modal Data Fusion

### 4.1 Gravity Inversion (Mass Deficit Detection)

GeoRisk uses Physics-Informed Neural Networks (PINNs) to invert Bouguer gravity anomalies. A subsurface void creates a **negative density contrast** detectable as a gravity low.

**Physics Layer**: Parker's Oldenburg formula (spectral domain)
```
F[g] = 2π G exp(-|k|z₀) F[ρ] × thickness
```

**Modifications for Sinkhole Detection**:
- Shallower depth estimate: 50m vs. 200m for minerals
- Void bias: Penalize positive density in loss function
- Focused on mass deficits (negative anomalies)

### 4.2 InSAR Velocity Analysis

Line-of-Sight (LOS) velocity from satellite interferometry provides:
1. **Vertical subsidence rate** (primary precursor)
2. **Horizontal strain** (critical for salt domes)
3. **Temporal acceleration** (exponential vs. logarithmic trend)

**3D Decomposition** (for salt dome regions):
Using ascending and descending passes:
```
Vertical = (LOS_asc + LOS_desc) / (2 cos θ)
East-West = (LOS_asc - LOS_desc) / (2 sin θ)
```

### 4.3 Spectral Indices (6-Month Lead Time)

Research from Poland (2024) detected spectral "structural breaks" 6 months before sinkhole formation:

| Index | Formula | Sinkhole Signal |
|-------|---------|-----------------|
| NDVI | (NIR - Red) / (NIR + Red) | Drop = root shear; Spike = water pooling |
| MI | (NIR - SWIR) / (NIR + SWIR) | Anomaly = groundwater rebound |

**Detection**: High localized variance compared to neighbors

### 4.4 Geological Proxies

Terrain-derived susceptibility factors:

| Proxy | Description | Risk Logic |
|-------|-------------|------------|
| Convergence Index | Flow concentration metric | High CI = water pooling = dissolution |
| Topographic Wetness (TWI) | ln(catchment / tan slope) | High TWI = saturation zones |
| Terrain Ruggedness (TRI) | Elevation roughness | Anomalous roughness = active raveling |

---

## 5. Algorithmic Filtering

### 5.1 DBSCAN Clustering

**Problem**: Single-pixel anomalies are typically noise (speckle, phase error).

**Solution**: DBSCAN (Density-Based Spatial Clustering) filters spatially coherent signals:
- A true precursor affects a **continuous patch** of ground
- Parameters: MinPts = 5-10 pixels, ε = 20-50 meters

**Result**: Reduces false positives by 80%+ in noisy urban environments.

### 5.2 Trend Analysis (Acute vs. Chronic)

| Pattern | Temporal Trend | Diagnosis |
|---------|----------------|-----------|
| Logarithmic decay | Fast→Slow | Benign consolidation |
| Exponential acceleration | Slow→Fast | **VOID PROPAGATION** |
| Linear stable | Constant | Monitoring zone |

**Implementation**: Rolling 6-month velocity trend with 50% acceleration threshold.

---

## 6. Risk Scoring

### Composite Score (0-100)

```
Composite = w₁×Velocity + w₂×Density + w₃×Geological + w₄×Proximity
```

**Florida Karst Weights**:
- Velocity: 35%
- Density: 25%
- Geological: 20%
- Proximity: 15%
- Spectral: 5%

### Risk Tiers (Insurance Classification)

| Score Range | Tier | Recommendation |
|-------------|------|----------------|
| 0-20 | Minimal | Annual review |
| 21-40 | Low | Semi-annual review |
| 41-60 | Moderate | Quarterly InSAR |
| 61-80 | Elevated | Monthly monitoring + ground inspection |
| 81-100 | Severe | **Immediate ground-truth required** |

---

## 7. Horizontal Strain (Salt Dome Protocol)

### The Bayou Corne Precedent

The 2012 Bayou Corne sinkhole displayed **260mm horizontal displacement** one month before collapse —  lateral "pinching" as surrounding rock moved toward the failing cavity.

### Implementation

For high-value assets in salt dome regions:
1. Mandate ascending + descending InSAR passes
2. Decompose to East-West strain component
3. Flag high horizontal gradients as "911-level" alerts

---

## 8. Validation & Confidence

### Literature Validation

| Study | Location | Precursor Detected | Accuracy |
|-------|----------|-------------------|----------|
| Oliver-Cabrera 2022 | Florida | 1 month - 1.7 years | High (X-band) |
| Konya Basin 2025 | Turkey | 65.2% of 2020-2024 sinkholes predicted | High |
| Bayou Corne 2014 | Louisiana | 1 month (horizontal) | Confirmed |
| Dead Sea 2024 | Israel | Days to 5 years | Moderate |

### Error Sources

- **False Positives**: Soil consolidation, thermal expansion → Mitigated by DBSCAN + RF
- **False Negatives**: Rapid brittle failure in Florida → Mitigated by X-band monitoring
- **Temporal Aliasing**: Fast motion between passes → Use higher revisit satellites

---

## 9. References

See `Deterministic-Geohazard-Forecasting.txt` for the complete bibliography (32 sources, 2020-2025).

Key papers:
1. Oliver-Cabrera et al. (2022) - Florida InSAR detection
2. Bayou Corne (2014) - Horizontal precursor validation
3. Konya Basin (2025) - ML susceptibility mapping
4. Dead Sea studies (2024) - Viscoelastic modeling

---

## 10. Commercial Application

### Data Products for Insurance

1. **Property Risk Reports**: Per-asset scoring with monitoring recommendations
2. **Portfolio Analytics**: Aggregate exposure and concentration risk
3. **Parametric Triggers**: Satellite-verified, oracle-ready thresholds
4. **API Integration**: Real-time risk queries for policy systems

### Pricing Model

- Per-property assessment
- Portfolio subscription
- Parametric trigger oracle fees
- Enterprise API access

---

*Document Version: 1.0 | Last Updated: December 2024*
