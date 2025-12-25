# GeoRisk 🌍⚠️

## AI-Powered Sinkhole Detection & Risk Assessment for Insurance

**Transform geohazard data into actionable insurance intelligence.**

---

## 🎯 What We Do

GeoRisk uses Physics-Informed Neural Networks (PINNs) combined with multi-modal satellite data fusion to **detect sinkhole precursors** and generate **insurance-grade risk assessments**.

### Our Core Value Proposition

| Traditional Insurance | GeoRisk Approach |
|----------------------|------------------|
| Probabilistic zone mapping | **Deterministic early warning** |
| Annual premium adjustments | **Real-time risk monitoring** |
| Post-event claims | **Pre-collapse intervention** |
| Static risk scores | **Dynamic velocity tracking** |

---

## 📊 Data Products for Insurance Companies

### 1. **Property Risk Reports** (Per-Asset)
- Individual property sinkhole risk score (1-100)
- Deformation velocity (mm/year)
- Distance to known sinkholes
- Geological susceptibility factors
- Recommended monitoring tier

### 2. **Portfolio Risk Analytics** (Batch)
- Aggregate exposure mapping
- Concentration risk identification
- Seasonal risk factors
- Claims prediction models

### 3. **Parametric Trigger Data**
- Automated deformation thresholds
- Satellite-verified trigger conditions
- Third-party oracle-ready outputs
- Settlement acceleration data

### 4. **Monitoring Dashboards**
- Real-time deformation tracking
- Alert notifications
- Historical trend analysis
- API integration for policy systems

---

## 🔬 Scientific Foundation

Based on peer-reviewed research (2020-2025) synthesized from:
- InSAR time-series analysis
- Microgravity surveys
- Multi-spectral vegetation indices
- Machine learning classification

### Validated Thresholds by Region

| Region | Alert Threshold | Critical Threshold | Sensor |
|--------|----------------|-------------------|--------|
| **Florida (Karst)** | -3 mm/yr | -6 mm/yr | X-band InSAR |
| **Texas (Salt Dome)** | -10 mm/yr | -20 mm/yr | Sentinel-1 |
| **Dead Sea** | -15 mm/yr | -20 mm/yr | Sentinel-1 |
| **Konya Basin** | -15 mm/yr | -30 mm/yr | InSAR + ML |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                             │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  InSAR Data  │ Gravity Data │ Spectral     │ Geological         │
│  (Sentinel)  │ (USGS/GRACE)│ (Sentinel-2) │ (USGS Lithostratigraphy) │
└──────────────┴──────────────┴──────────────┴────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                          │
├──────────────────────────────────────────────────────────────────┤
│  1. PINN Gravity Inversion → Mass Deficit Detection              │
│  2. InSAR Velocity Analysis → Deformation Tracking               │
│  3. DBSCAN Clustering → Noise Reduction / Anomaly Isolation      │
│  4. Random Forest Classification → Geological Context Filtering  │
│  5. Acceleration Analysis → Chronic vs Acute Differentiation     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RISK SCORING ENGINE                          │
├──────────────────────────────────────────────────────────────────┤
│  • Velocity Score (weighted by geology)                          │
│  • Acceleration Score (exponential vs decay curve)               │
│  • Proximity Score (distance to known sinkholes)                 │
│  • Geological Susceptibility Score                               │
│  • Spectral Anomaly Score (NDVI/MI variance)                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      INSURANCE DATA PRODUCTS                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ Property     │ Portfolio    │ Parametric   │ API                │
│ Risk Reports │ Analytics    │ Trigger Data │ Integration        │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## 📁 Project Structure

```
GeoRisk/
├── georisk/
│   ├── core/
│   │   ├── pinn_sinkhole_inversion.py    # Void-mode gravity inversion
│   │   ├── insar_processor.py            # InSAR velocity analysis
│   │   ├── risk_engine.py                # Multi-factor scoring
│   │   └── clustering.py                 # DBSCAN anomaly detection
│   ├── data/
│   │   ├── inputs/                       # Raw satellite data
│   │   └── outputs/                      # Processed risk data
│   ├── models/
│   │   └── pretrained/                   # Trained PINN models
│   ├── utils/
│   │   ├── data_fetcher.py               # Satellite data acquisition
│   │   ├── geological_proxies.py         # TWI, TRI, CI calculations
│   │   └── spectral_indices.py           # NDVI, MI processing
│   ├── api/
│   │   ├── insurance_reports.py          # Report generation
│   │   └── parametric_triggers.py        # Trigger system
│   └── docs/
│       └── methodology.md                # Scientific documentation
├── requirements.txt
├── config.yaml
└── run_pipeline.py                       # Main execution script
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/georisk.git
cd georisk

# Install dependencies
pip install -r requirements.txt

# Configure your region
cp config.example.yaml config.yaml
# Edit config.yaml with your target region

# Run the pipeline
python run_pipeline.py --region florida --mode batch
```

---

## 💰 Commercial Applications

### For Property & Casualty Insurers
- **Underwriting Enhancement**: Score new policies before binding
- **Portfolio Management**: Identify concentration risk
- **Claims Prediction**: Early warning before catastrophic losses

### For Reinsurers
- **Treaty Pricing**: Data-driven layer pricing
- **Accumulation Management**: Real-time exposure tracking
- **Retrocession Analysis**: Granular cedant portfolio insights

### For Parametric Insurance
- **Trigger Definition**: Satellite-based objective triggers
- **Oracle Data**: Third-party verifiable deformation data
- **Fast Settlement**: Automated payout triggers

---

## 📜 License

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

This project is licensed for non-commercial use only. Commercial use of this 
code, derived models, or generated risk data requires a separate commercial license.

For commercial licensing inquiries, contact the authors.

---

## ⚠️ Disclaimer

GeoRisk provides risk assessment data based on satellite observations and machine learning models. While our methodology is grounded in peer-reviewed science, this data should be used as one input among many in insurance decision-making. We do not guarantee prediction accuracy and are not liable for individual sinkhole events.

---

*Built with AI assistance • December 2024*
