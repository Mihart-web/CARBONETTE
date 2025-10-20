[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Backend](https://img.shields.io/badge/backend-FastAPI-orange)
![Engine](https://img.shields.io/badge/engine-Carbonette-blueviolet)


# CARBONETTE CNT/HNT pipeline v1.0.0
Author: M. Vengher
Purpose: Automated spectral analysis for CNT/HNT feature detection (5–20 µm)
Methods: quick-look, SAFE masks, LSF injection-recovery, core–wide coherence,
template matching (Chen & Li 2019/2020), slope & shape coherence tests.
# 🔬 CARBONETTE: Automated CNT/HNT Spectral Detection Pipeline
Carbonette scans infrared spectra and highlights where carbon nanotube (CNT) and hydrogenated nanotube (HNT) bands might be hiding. It’s a careful filter: it doesn’t declare discoveries, it shows where the data deserve a closer look. This analysis engine is based on publicly available Spitzer spectra (IRSA Enhanced Products) and tests the infrared bands predicted for carbon nanotubes by Chen & Li (2019, 2020). The authors of those studies are not affiliated with or responsible for this project.
(JWST analyzer extension - work in progress)

## 🎯 Overview
CARBONETTE is a sophisticated automated pipeline for detecting Carbon NanoTube (CNT) and Hydrogenated NanoTube (HNT) absorption features in infrared astronomical spectra. The system processes data directly from NASA's IRSA Enhanced Products Catalog and implements a rigorous multi-validation framework to ensure scientifically robust detections.

## ✨ Comprehensive Validation Framework

### 📊 **Primary Detection Metrics**
- **Signal-to-Noise Ratio (S/N)** calculation with proper error propagation
- **Δχ² Significance Testing** for feature detection confidence
- **Equivalent Width (EW)** and **Mean Optical Depth (τ)** measurements
- **Multi-threshold Classification**: 
  - YES (detection): S/N ≥ 3.0σ AND √Δχ² ≥ 3.0σ
  - MARGINAL: 2.0σ ≤ S/N < 3.0σ
  - NO: S/N < 2.0σ

### 🛡️ **Contamination Avoidance (SAFE System)**
- **Literature-informed anchor regions** for continuum fitting
- **PAH contamination checks** at 11.2-11.3μm and 12.7μm
- **Water vapor avoidance** at 5.3μm regions
- **Standard vs SAFE band comparison** with automatic promotion rules
- ** Masks and anchors from literature

### 📏 **Instrumental Validation**
- **Line Spread Function (LSF) resolution checks**
- **Width-to-LSF ratio analysis** ensuring features are instrumentally resolved
- **Rebinning consistency** across different spectral resolutions (R=1, R=2)

### 🧪 **Completeness & Reliability Testing**
- **Injection-Recovery analysis** with 200 bootstrap iterations
- **Multiple depth testing** (0.5%, 1.0%, 2.0% absorption depths)
- **Recovery rate calibration** and probability estimation
- **Detection completeness mapping** across parameter space

### 🔄 **Stability & Robustness Verification**
- **Core-edge sensitivity analysis** (±0.05μm shifts)
- **Jackknife resampling** for uncertainty estimation
- **Continuum model comparison** (linear vs quadratic fitting)
- **Rebinning consistency checks** across resolutions

### 🎯 **Physical Validation**
- **Template matching** against established CNT/HNT models (Chen & Li 2020)
- **Best-fit identification** with Δχ² minimization
- **Slope asymmetry analysis** for feature shape coherence
- **Cross-band correlation** testing

### 📈 **Advanced Statistical Framework**
- **Benjamini-Hochberg FDR correction** for multiple hypothesis testing
- **False Discovery Rate control** at α=0.05
- **Dynamic p-value thresholding** based on number of tests
- **Matched filtering confirmation** as independent validation

### 🚫 **Quality Control Flags**
- **PAH leakage detection** at 11.2μm
- **Anchor point sufficiency** checks (minimum 2 required)
- **Core data point validation** (minimum 3 required)
- **Global spectral masking** of known emission lines

## 📋 **Output Generation**

### 📄 **Professional Reporting**
- **Publication-ready PDF reports** with full analysis summaries
- **LaTeX source files** for custom formatting
- **CSV data exports** for further analysis
- **Multi-format plots** with residual panels

### 🔍 **Comprehensive Validation Summaries**
- **Band-by-band certification tables**
- **Standard vs SAFE comparison reports**
- **Core vs Wide confirmation analysis**
- **Injection-recovery completeness maps**

## 🚀 **Technical Features**

### 🌐 **Data Integration**
- **Direct IRSA Enhanced Products Catalog access**
- **Automated spectral data retrieval**
- **Flexible input formats** (wavelength, flux, error, flag)
- **Intelligent data parsing** with error handling

### ⚙️ **Configurable Analysis**
- **User-defined detection thresholds**
- **Customizable band definitions**
- **Adjustable rebinning factors**
- **Flexible anchor region specification

### 🔧 **Robust Implementation**
- **Error propagation** throughout all calculations
- **Outlier-resistant** continuum fitting
- **Graceful degradation** with insufficient data
- **Comprehensive logging** and progress tracking

## 🎪 **Final Detection Certification**

A **"YES (robust)"** detection requires:
```
✅ Primary S/N ≥ 3.0σ AND √Δχ² ≥ 3.0σ
✅ SAFE variant confirmation (where applicable)  
✅ LSF resolution adequacy (width/LSF > 5)
✅ Injection-recovery rate ≥ 80% at relevant depth
✅ Stability tests passed (rebin, edge, jackknife, poly2)
✅ Template matching consistency
✅ FDR-corrected significance
```

## 📚 **Scientific Foundation**

The pipeline implements methodologies from established astronomical spectroscopy literature, with particular attention to:
- Continuum placement in uncontaminated spectral regions
- Statistical significance validation
- Instrumental effect quantification
- Contamination avoidance strategies

---

**CARBONETTE represents a comprehensive, validation-focused approach to spectral feature detection, implementing professional-grade astronomical analysis techniques in an automated, reproducible framework.**

Tests made to build this pipeline: All objects analyzed with this pipeline are
drawn from the IRSA Spitzer Enhanced Products archive.

