# New Moscow Real Estate Valuation Stacking Framework

An end-to-end, production-grade gradient boosting ensemble pipeline that achieved a **Top 3 Podium Finish (#3)** in the [New Moscow Housing Valuation Challenge Leaderboard](https://www.kaggle.com/competitions/ne-vsos-ii-new-moscow/leaderboard).

## 📊 Final Performance Results

| Phase / Iteration | Public Leaderboard RMSE | Private Leaderboard RMSE (Final) | Technical Strategy Shift |
| :--- | :--- | :--- | :--- |
| Baseline | 303.14 | *Overfitted* | Vanilla LightGBM Gradient Boosting |
| Mid-Challenge | 301.06 | 302.40 | Single-Target Robust Huber Loss Pipeline |
| **Final Ensemble** | **300.54** | **299.10** 🔥 | **Multi-Objective Target Stacking ($y_{\text{ppm}}$ + $y_{\text{raw}}$)** |

## 💡 Core Innovations

### 1. Multi-Objective Stacking Architecture
Instead of optimizing solely on the raw absolute pricing target (`y_raw`), this framework trains base regressors (CatBoost, LightGBM, XGBoost) on two distinct mathematical scales concurrently:
* **Horizon A:** Price-per-Square-Meter ratio ($y_{\text{ppm}}$)
* **Horizon B:** Total Absolute Property Budget ($y_{\text{raw}}$)

By exposing out-of-fold predictions from both dimensions across a 5-fold cross-validation grid, a meta-regressor (`RidgeCV`) learns to dynamically balance density valuation against macro volume. This dual-axis target visibility minimized structural bias and dramatically improved generalization performance against the final leaderboard split.

### 2. Leak-Proof Out-Of-Fold Target Encoding
To capture deep spatial signals from high-cardinality categorical dimensions (such as structural building IDs and localization hashes) without causing catastrophic target leakage, features are computed using an isolated, out-of-fold cumulative calculation engine.

## 📂 Repository Architecture

The project codebase is modularized according to industry standard engineering patterns:

```text
├── src/
│   ├── __init__.py
│   ├── engineering.py     # Data pipeline transformations & leak-proof target encoding grid
│   └── models.py          # Parallel multi-objective base learners & regularized stacking layer
├── main.py                # Production entry point for inference generation
├── requirements.txt       # Environment software dependencies
└── README.md              # Project executive overview
