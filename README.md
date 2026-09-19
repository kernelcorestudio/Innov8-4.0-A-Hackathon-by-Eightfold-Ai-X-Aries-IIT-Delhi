# 🏆 The Corporate Heist — Innov8 4.0

> **Hackathon organized by Eightfold.ai × ARIES, IIT Delhi**  
> **Team:** KC STUDIO  
> **Team Members:** Jay Negi, Tarun Ruwali, Faizan Ali Ansari, Ankush Singh Rawat, Karan Nayal  

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML%20Framework-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Recall%40150-0.393-brightgreen.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-~4s%20(CPU)-blueviolet.svg)]()
[![Format](https://img.shields.io/badge/Format%20Check-Passed-brightgreen.svg)]()

---

## 📌 Executive Summary

**The Corporate Heist** challenged participants to break through legacy hiring biases and rank candidates from Nightingale's applicant pool ("The Vault", `test.csv`) to surface the true **Top 500** hires for the current hiring cycle. 

Our solution combines:
1. **Systematic Free-Text Parsing & Normalization (`code/clean_utils.py`)**: Extracting clean numerical representations across messy, recruiter-typed fields with mixed units and inconsistent scales.
2. **Pedigree-Debiased Supervised Learning (`code/main.py`)**: A LightGBM regression model trained on historical performance (`train.csv`) utilizing pure competence, productivity, and retention indicators while explicitly stripping out historical pedigree biases (college tier, metro cities, brand-name employers, referral channels).
3. **Role-Normalized Z-Scores**: Calibrating technical assessments and aptitude scores within candidate job roles (`applied_role`).
4. **Cycle-Specific Heuristics & Grounding**: Rewarding sustained public code contributions (new in this cycle) and applying minor leadership alma-mater adjustments.
5. **Integrity, Fraud Detection & Deduplication (`code/integrity_utils.py`)**: Hard exclusion of planted traps (title inflation, inconsistent/fabricated CVs, >60-day notice periods) and fuzzy deduplication of candidates submitted multiple times.

---

## 👥 Team: KC STUDIO

| Member Name | Role / Focus |
| :--- | :--- |
| **Jay Negi** | Data Cleaning & Feature Engineering |
| **Tarun Ruwali** | Model Architecture & Hyperparameter Tuning |
| **Faizan Ali Ansari** | Integrity Analysis & Deduplication Algorithms |
| **Ankush Singh Rawat** | Pipeline Integration & Validation |
| **Karan Nayal** | Domain Heuristics & Documentation |

---

## 🧠 Methodology & Architecture

### 1. Robust Data Cleaning & Parsing (`clean_utils.py`)

Recruiter input is notoriously messy. Every free-text column was transformed using custom deterministic regex parsers:

| Raw Column | Input Examples | Parsed Scale / Rule |
| :--- | :--- | :--- |
| `total_experience` | `"8+ yrs"`, `">20"`, `"19 months"`, `"17.6 years"` | Years ($e \in \mathbb{R}_{\ge 0}$) |
| `aptitude_score` | `"5.7/10"`, `"86%"`, `"7.3"` | Scaled to $0 - 10$ |
| `last_rating` | `"5/5"`, `"Outstanding"`, `"3"`, `"New joiner - not rated"` | $1 - 5$ scale via text mapping / numerals |
| `notice_period` | `"Immediate"`, `"2 months"`, `"Serving notice - 45 days"` | Calendar days (e.g. `Immediate` $\to 0$, `2 months` $\to 60$) |
| `current_ctc` / `expected_ctc` | `"58.3 LPA"`, `"9.85 Cr"`, `"₹41,20,000"`, `"1692000"` | Normalized to INR Lakhs |
| `technical_assessment` | `"49/100"`, `"0.43"`, `"85"` | Normalized to $0 - 100$ scale |
| `awards` | `"-"`, `"0"`, `"Yes - Spot Award"`, `"1 award"` | Non-negative integer count |
| `kpi_met`, `overtime_history` | `"Y"`, `"yes"`, `"1"`, `"TRUE"` vs. `"N"`, `"no"`, `"0"`, `"FALSE"` | Binary indicator $\{0, 1\}$ |
| `public_code_contributions` | `"~5"`, `"12 merged PRs"`, `"not tracked"` | Count; `NaN` if not tracked |
| `institute` | `"IIT-D"`, `"I.I.T. Delhi"`, `"IITB"`, `"IIT Kanpur"` | Canonicalized with regex normalization |

We also engineered:
- **`years_since_grad`** $= 2026 - \text{graduation\_year}$
- **`age_at_grad`** $= \text{age} - \text{years\_since\_grad}$
- **`exp_gap`** $= \text{exp\_years} - \text{years\_since\_grad}$
- **`ctc_ratio`** $= \text{clip}\left(\frac{\text{expected\_ctc\_lakhs}}{\text{current\_ctc\_lakhs}}, 0, 10\right)$
- **`tech_z` & `apt_z`**: Z-scores computed within each `applied_role`.

---

### 2. Debiased Model Training (`main.py`)

To eliminate historical committee biases, features associated with pedigree (college prestige, current city, previous employer brand, recruitment channel, degree classification) were completely omitted.

- **Algorithm**: `LGBMRegressor`
- **Hyperparameters**: `n_estimators=400`, `learning_rate=0.03`, `num_leaves=31`, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
- **Target**: `post_hire_score` from `train.csv`.
- **Core Features**:
  ```python
  CORE_FEATURES = [
      'tech_assessment_clean', 'aptitude_clean', 'rating_clean', 'kpi_met_clean',
      'awards_clean', 'trainings_last_year', 'training_hours', 'overtime_clean',
      'enrolled_clean', 'exp_years', 'num_employers', 'last_job_change_clean',
      'notice_days', 'ctc_ratio', 'tech_z', 'apt_z'
  ]
  ```

---

### 3. Cycle Adjustments & Composite Scoring

Standardized model predictions are augmented with domain-verified adjustments:
$$\text{final\_score}_i = z_i + 0.35 \cdot \text{sustained\_coder}_i + 0.10 \cdot \text{some\_coder}_i + 0.15 \cdot \text{leadership\_college}_i$$

- **`sustained_coder`** ($\ge 24$ contributions/year): Fast-tracks active open-source contributors (new signal in this cycle).
- **`some_coder`** ($1 - 23$ contributions/year): Small positive recognition.
- **`leadership_college`**: Small additive boost ($+0.15$) for top leadership IIT alma maters (Delhi, Bombay, Kanpur, Madras) acting as a tie-breaker.

---

### 4. Hard Filters & Integrity Screening (`integrity_utils.py`)

A four-stage audit removes fraudulent, exaggerated, or ineligible profiles:

| Filter | Logic / Rule | Candidates Excluded in Vault | Rationale |
| :--- | :--- | :---: | :--- |
| **Title Inflation** | Senior title (`Head of`, `VP`, `Director`, `Chief`) with $<6$ years experience | **65** | Planted trap (0% in Archive, 0.65% in Vault). |
| **Fabrication Score** | $\ge 2$ composite red flags (abnormal graduation age, experience exceeding career span, negative experience, employer-hopping ratio, or simultaneous max scores on every axis) | **102** | Identified as "too good to be true" / inconsistent. |
| **Long Notice Period** | `notice_days > 60` | **208** | Strict 2-month joining constraint for this cycle. |
| **Duplicate Profiles** | Grouped by canonical name, graduation year, and institute, plus RapidFuzz name matching ($\ge 90$) | **125** rows removed | Single real individual entered multiple times; only the top-scoring profile is preserved. |

**Final Eligible Pool:** **9,502 / 10,000** candidates evaluated for the Top 500 shortlist.

---

## 📊 Validation & Benchmark Results

We validated our pipeline on the historical Ledger dataset (`dev.csv` + `dev_winners.csv`):
- **Recall@150**: **`0.393`** (59/150 ground-truth winners recovered in top 150).
- **Baseline Lift**: **~8× improvement** over random chance (`~0.05`).
- **No-Filter Baseline**: `0.387` $\to$ hard filters and deduplication deliver measurable performance gains.

---

## 📁 Repository Structure

```
.
├── code/
│   ├── clean_utils.py          # Robust text parsers & feature derivation
│   ├── integrity_utils.py      # Fabrication scoring & duplicate clustering
│   └── main.py                 # Training, scoring, filtering & ranking pipeline
├── submission.csv              # Top 500 final shortlist (rank 1 to 500)
├── documentation.pdf           # 4-page official competition documentation
├── documentation-KC STUDIO.pdf # Solution documentation by Team KC STUDIO
├── check_format.py             # Official submission format verification script
├── dev.csv                     # Ledger validation dataset
├── dev_winners.csv             # Ground truth winners for dev dataset
├── train.csv                   # Archive training dataset
├── test.csv                    # Vault test dataset
└── README.md                   # Comprehensive solution overview
```

---

## 🚀 Reproduction & Execution Guide

### 1. Prerequisites
- Python **3.11**
- Required libraries:
  ```bash
  pip install numpy pandas lightgbm rapidfuzz
  ```

### 2. Run the Full Pipeline
Run `code/main.py` from the project root:
```bash
python code/main.py
```
**Expected Output:**
```
[DEV VALIDATION] recall@150 = 0.393  (pool size after filters/dedup = 2933 / 2999)
[DEV VALIDATION] recall@150 without hard filters/dedup = 0.387
Excluded: title-inflated=65, fabricated=102, long-notice=208, duplicate-rows-removed=125
Final eligible pool: 9502 / 10000
Wrote submission.csv with 500 rows
```
*Total execution runtime: **~4 seconds** on standard CPU.*

### 3. Verify Submission Format
Validate the generated shortlist with the official format checker:
```bash
python check_format.py submission.csv test.csv
```
**Output:**
```
OK - submission.csv is correctly formatted.
```

---

## 📦 Submission Archive Checklist

For submission on Unstop, create `<team_name>.zip` containing:
```
<team_name>.zip
├── submission.csv
├── documentation.pdf
└── code/
    ├── main.py
    ├── clean_utils.py
    └── integrity_utils.py
```
*(Raw data files `train.csv`, `test.csv`, and `dev.csv` are omitted from the final archive per competition rules).*

---

## 📜 Declarations & Compliance
- **Permitted Libraries**: Used only `numpy`, `pandas`, `lightgbm`, and `rapidfuzz` from Appendix A.
- **Hardware Limits**: Executed entirely on CPU, requiring $< 1$ GB RAM and running in $\sim 4$ seconds (well under the 4 GB, 5-minute limits).
- **AI Tool Usage**: Claude (Anthropic) assisted with code structuring, regex design, and documentation drafting. All choices, thresholds, and numbers were verified against the provided datasets.
