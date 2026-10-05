# eThekwini Local Government Election Analytics & Forecasting

**Student Number:** 22420947  
**Surname & Initials:** Gumede SN 
**Project:** South African Local Government Election Analytics  
**Municipality:** eThekwini Metropolitan Municipality  
**Language:** Python  
**Dashboard:** Streamlit  

---
> **Important Data Provenance Note:**  
> The four CSV files listed above are the **only original source datasets** used for this project. CSV files located inside the `eThekwini_Streamlit_Dashboard` folder are **not additional datasets and are not independently collected or fabricated data**. They are derived analytical outputs produced from the project's data-processing, modelling and forecasting workflow and are stored only so that the Streamlit dashboard can efficiently display the results
## 1. Project Overview

This project implements an end-to-end Machine Learning Life Cycle (MLLC) for analysing historical election data and producing evidence-based election estimates for the eThekwini Metropolitan Municipality.

The solution covers data acquisition, data understanding, cleaning and preprocessing, exploratory data analysis (EDA), feature engineering, model development, validation, evaluation, forecasting and deployment through an interactive Streamlit dashboard.

The analysis addresses four principal outputs:

1. Projected leading party and aggregate metro-level vote share.
2. Projected leading party in three selected wards.
3. Projected vote totals and/or vote shares for relevant parties.
4. Projected voter turnout for eThekwini.

Predictions are presented as **model estimates rather than statements of electoral certainty**.

---

## 2. Data Sources

All original election datasets were obtained from the **Electoral Commission of South Africa (IEC)**.

### Original Source Data

| Dataset | Purpose |
|---|---|
| `2011 LOCAL GOVERNMENT.csv` | Historical Local Government Election data |
| `2016 LOCAL GOVERNMENT.csv` | Historical Local Government Election data |
| `2021 LOCAL GOVERNMENT.csv` | Most recent Local Government Election training/history data |
| `2024 NATIONAL.csv` | Recent electoral signal, including parties such as uMkhonto weSizwe (MK) |

**IEC Municipal Election Results:**  
https://results.elections.org.za/home/downloads/me-results

**IEC National and Provincial Election Results:**  
https://results.elections.org.za/home/Downloads/NPE-Results

The 2024 National Election is **not treated as directly equivalent to a Local Government Election**. It is incorporated as supplementary recent electoral evidence, particularly because MK has no historical LGE observations in the 2011–2021 datasets.

---

## 3. Repository Structure

```text
2026-tp-exam-22420947/
│
├── 2026_Final_TP2_22420947.ipynb
│
├── 2011 LOCAL GOVERNMENT.csv
├── 2016 LOCAL GOVERNMENT.csv
├── 2021 LOCAL GOVERNMENT.csv
├── 2024 NATIONAL.csv
│
├── eThekwini_Streamlit_Dashboard/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── historical_party_shares.csv
│   ├── metro_stats.csv
│   ├── npe2024.csv
│   ├── forecast_party_table.csv
│   ├── forecast_wards.csv
│   ├── validation_scores.csv
│   └── model_metrics.json
│
└── README.md
