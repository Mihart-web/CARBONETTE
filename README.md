# CARBONETTE CNT/HNT pipeline v1.0.0
Author: M. Vengher
Purpose: Automated spectral analysis for CNT/HNT feature detection (5–20 µm)
Methods: quick-look, SAFE masks, LSF injection-recovery, core–wide coherence,
template matching (Chen & Li 2019/2020), slope & shape coherence tests.
# 🔬 CARBONETTE: Automated CNT/HNT Spectral Detection Pipeline
Purpose Carbonette scans infrared spectra and highlights where carbon nanotube (CNT) and hydrogenated nanotube (HNT) bands might be hiding. It’s a careful filter: it doesn’t declare discoveries, it shows where the data deserve a closer look. This analysis engine is based on publicly available Spitzer spectra (IRSA Enhanced Products) and tests the infrared bands predicted for carbon nanotubes by Chen & Li (2019, 2020). The authors of those studies are not affiliated with or responsible for this project.

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


CARBONETTE  PIPELINE — INTERPRETATION GUIDE

Tests made to build this pipeline: All objects analyzed with this pipeline are
drawn from the IRSA Spitzer Enhanced Products archive.

This section provides a quick legend explaining the meaning of each diagnostic
table and metric in the CNT/HNT pipeline. It is intended for reviewers,
students, or collaborators to interpret results correctly and responsibly.

------------------------------------------------------------
1. Quick-look table
------------------------------------------------------------
• S/N (signal-to-noise)  — Integrated significance of the deficit (negative) or
  emission (positive) within the defined band.
• √Δχ² (zsig)  — Equivalent Gaussian significance derived from Δχ².
• EW (equivalent width)  — Band strength in μm. Negative = absorption, positive = emission.
• τ̄ (mean optical depth) — Average optical depth over the band.
• verdict  — "YES" indicates a statistically significant feature; "NO" = none.

Interpretation:
    YES (detection) → statistically significant spectral feature.
    YES (robust)    → confirmed by multiple independent checks.
    NO              → no significant feature.
    MARGINAL        → weak, possibly real but below threshold.

------------------------------------------------------------
2. SAFE / water-safe / PAH-safe comparison
------------------------------------------------------------
Tests whether the same feature persists when nearby PAH or water bands are
masked.  "YES (robust)" means the signal survives all safety masks.

------------------------------------------------------------
3. Core vs Wide cross-confirmation
------------------------------------------------------------
Each band is measured in two versions (core and wide) to check consistency.
Verdict:
    IT FITS    → same sign and amplitude trend (consistent matching).
    MARGINAL   → slightly inconsistent.
    NO         → opposite sign or incoherent shape.

------------------------------------------------------------
4. LSF and Injection–Recovery tests
------------------------------------------------------------
• width/LSF ≥ 1 → the band is resolved relative to the instrument line spread.
• rec_rate ≥ 0.8 → artificial injections of similar depth are recovered ≥80% of times.
These verify that the signal is neither noise nor instrumental.

------------------------------------------------------------
5. Extra Verification
------------------------------------------------------------
Additional robustness tests:
    – Rebin(1,2): stability across spectral resolutions.
    – Edge ±0.05: shift of integration limits.
    – Jackknife: stability when excluding subsets of points.
    – Poly2 vs Linear: stability against continuum model choice.
All must pass (OK) for a "CONFIRM_STRONG" classification.

------------------------------------------------------------
6. Template Matching (Chen & Li)
------------------------------------------------------------
Observed residuals are compared with synthetic CNT/HNT spectra predictions from
Tao Chen & Aigen Li (2019, "Synthesizing carbon nanotubes in space").
• z (√Δχ²) > 5 → strong morphological match.
• best_template → which CNT/HNT family fits best (neutral, cation, HNT...).
• alpha → scaling factor; positive = absorption, negative = emission.

------------------------------------------------------------
7. Slope Check (polarization/slope test)
------------------------------------------------------------
Left vs right continuum slopes (±0.5 μm from band center) are compared.
    CHECK → slope changes significantly (real continuum distortion).
    OK    → consistent with noise.

------------------------------------------------------------
8. Shape Coherence (spectral correlation)
------------------------------------------------------------
Normalized correlation r between bands (10.7–12.9–17 µm):
    r ≈ +1 → similar profile shape (same carrier species).
    r ≈ −1 → mirrored/opposite shape (e.g., emission vs absorption).
    |r| < 0.5 → unrelated.
p-value near zero → correlation statistically significant.

AND MUCH MORE - You can find everything in the reports, pdf,csv,png ecc.
------------------------------------------------------------
9. Interpretation and Caution
------------------------------------------------------------
These diagnostics indicate consistency with CNT/HNT-like features,
not direct proof of carbon nanotubes or nanostructures.
Treat outputs as candidates pending verification via
  – higher-resolution spectroscopy (e.g., JWST/MIRI),
  – polarization or imaging,
  – laboratory spectra comparison.
No discovery claims are made; results guide further study.
📊 Table A — Dust, Molecules, Nanostructures (CNT, PAH, Fullerenes, Ices, etc.)
Paper (year, authors)	Species / Feature	Band (µm / Å / GHz / THz)	Type	Notes
  Megías et al. (2025)	H2O ice	≈3, ≈6 µm	Absorption	Stretch & bending
  CO2 ice	4.27, 15.2 µm	Absorption	
  CO ice	4.67 µm	Absorption	
  CH4 ice	7.7 µm	Absorption	
  NH3 ice	2.96 µm	Absorption	
  CH3OH ice	3.53, 9.7 µm	Absorption	
  Yoon et al. (2024) – PRIMA/FIRESS PAH	PAH (qPAH models)	3.3, 6.2, 7.7, 11.2 µm	Emission	qPAH = 0.5%, 1.8%, 3.8%
  Chen & Li (2020) – CNTs in Space	CNT neutral (NT)	5.3, 7.0, 8.8, 9.7, 10.9, 14.2, 16.8 µm	Absorption	Predicted IR bands
  CNT+ (NT+)	5.2, 7.1, 8.3, 9.2, 13.4, 16.7 µm	Absorption	
  HNT	3.3, 6.5, 8.2, 10.8, 13.0, 17.2 µm	Absorption	Hydrogenated CNTs
  HNT+	3.3, 6.6, 8.3, 10.6, 12.8, 17.2 µm	Absorption	Hydrogenated cationic CNTs
  C60	7.0, 8.45, 17.3, 18.9 µm	Emission	Fullerene
  C60+	6.4, 7.1, 8.2, 10.5 µm	Emission	Fullerene cation
  Graphene C24	6.6, 9.8, 20 µm	Absorption	Graphene sheet
  PAH (classic)	3.3, 6.2, 7.7, 8.6, 11.3, 12.7 µm	Emission	Aromatic bands
  Shuba et al. (2008) – CNT absorption	Metallic CNT (zigzag (9,0))	12.7 µm	Absorption	Geometric resonance
  Metallic CNT	4.9 µm	Absorption	
  Metallic CNT	3.2 µm	Absorption	Antenna-like
  Rai & Rastogi (2009) – Nanodiamonds	Nanodiamond H–C stretch	3.43, 3.53 µm	Emission	Hydrogenated
  Nanodiamond	3.47 µm	Absorption	Tertiary C–H
  Nanodiamond/graphite	2175 Å (0.2175 µm)	Absorption	UV bump
  Gavdush et al. (2025) – Ice Analogues III	CO ice	≈1.5 THz (200 µm)	Absorption	Vibrational
  CO2 ice	≈3.5 THz (85 µm)	Absorption	Vibrational
  CO2 ice	15–18 THz (~20 µm)	Absorption	Sidebands, porosity
  📊 Table B — Atomic and Ionic Lines (JWST/MIRI, PRIMA/FIRESS, etc.)
  Paper (year, authors)	Species / Feature	Band (µm)	Type	Notes
  Kastner et al. (2025) – JWST/NIRCam	[Fe II]	1.64 µm	Emission	Ionized
  H2	2.12 µm	Emission	Molecular
  H I (Brα)	4.05 µm	Emission	Hydrogen
  Fernández-Ontiveros et al. (2025) – PRIMA/FIRESS	[S IV]	10.5 µm	Emission	Ionized
  [Ne III]	15.6 µm	Emission	Ionized
  [O IV]	25.9 µm	Emission	Ionized
  [O III]	52, 88 µm	Emission	Ionized
  [N III]	57 µm	Emission	Ionized
  [N II]	122, 205 µm	Emission	Ionized
  Hermosa Muñoz et al. (2025) – JWST/MIRI (GATOS)	H2 0–0 S(2)	12.28 µm	Emission	Molecular
  H I (7–6)	12.37 µm	Emission	Hydrogen
  [Ne II]	12.81 µm	Emission	Ionized
  [Ar V]	13.10 µm	Emission	Ionized
  [Mg V]	13.52 µm	Emission	Ionized
  [Ne V]	14.32 µm	Emission	Ionized
  [Cl II]	14.37 µm	Emission	Ionized
  [Ne III]	15.56 µm	Emission	Ionized
  H2 0–0 S(1)	17.03 µm	Emission	Molecular
  [Ar VI]	11.6–11.76 µm	Emission	High excitation
  Setton et al. (2025) – ALMA SQuIGGŁE	CO(2–1)	230.538 GHz (~1.30 mm)	Emission	Molecular (ALMA)
  This page is explanatory; it does not claim detections. Use it as an interpretation key for Carbonette outputs.
