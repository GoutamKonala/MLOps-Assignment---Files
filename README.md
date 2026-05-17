# Predictive Maintenance Classification with Local MLOps

**Domain:** Industrial IoT / Manufacturing

**Task:** Multi-class failure type prediction

## Business Context

A heavy-equipment manufacturer runs 10,000+ machines on the shop floor.
Each machine continuously streams six sensor readings.
When a machine fails, production halts at a cost of 8-15 lakh per hour
of downtime.

This project builds a full local MLOps workflow to:

1. Validate incoming sensor data before it enters the pipeline (Pandera)
2. Train and track a multi-class failure classifier (MLflow)
3. Tune hyperparameters and register the best model (Optuna + MLflow Registry)
4. Monitor the deployed model for distributional shift (Evidently)
5. Explain why the model predicts a specific failure type (SHAP)

## Dataset Details

The dataset represents sensor readings from manufacturing equipment.
It has been pre-split into three CSVs for the assignment:

- `data/train.csv`: 6,993 labelled sensor readings
  (historical baseline used for training and validation)
- `data/current.csv`: 1,499 readings from the current stable production batch
- `data/stress.csv`: 1,499 readings from a heavy-load production period
  (valid but drifted data)

### Failure Classes

| Code | Name       | Description                |
| ---- | ---------- | -------------------------- |
| 0    | No Failure | Machine operating normally |
| 1    | TWF        | Tool Wear Failure          |
| 2    | HDF        | Heat Dissipation Failure   |
| 3    | PWF        | Power Failure              |
| 4    | OSF        | Overstrain Failure         |

## Setup Instructions

Ensure you have Python installed.
The required dependencies are listed in `requirements.txt`.
To install them, run:

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook `MLOps_Assignment_Lakshmi_Goutam_Reddy_Konala.ipynb`
to execute the full pipeline. Ensure MLflow tracking is active
(a local `mlflow.db` is configured in the notebook).

## Technologies Used

- Data Validation: `pandera`
- Experiment Tracking & Registry: `mlflow`
- Hyperparameter Tuning: `optuna`
- Drift Detection: `evidently`
- Model Explainability: `shap`
- ML Models: `xgboost`, `lightgbm`, `scikit-learn`
- Imbalanced Learning: `imbalanced-learn` (SMOTE)

## Key Findings and Conclusions

1. **Model Selection:**
   XGBoost was chosen as the final model with a Tuned Macro F1 of 0.7610.
   It outperformed simpler models because its gradient boosting architecture
   effectively handled the non-linear sensor relationships.

2. **Accuracy vs Macro F1:**
   Accuracy was highly misleading (98%+), hiding the model's struggle
   with rare classes. For manufacturing, missing one expensive failure
   (False Negative) is far more costly than a few extra checks on healthy
   machines.

3. **The TWF Problem:**
   The root cause of the low F1 (0.13) for TWF is data scarcity—there
   were only 30 real examples in the training set. To fix this, more
   real-world failure data must be collected rather than relying solely
   on synthetic SMOTE samples.

4. **Drift and Maintenance Schedule:**
   The stress batch showed that under high load, Tool wear and Torque drifted
   significantly. This implies maintenance schedules must be more aggressive
   during high-production periods to compensate for faster-than-expected wear.

5. **Actionable Recommendation:**
   - **Condition:** If Tool wear > 200 or Torque shifts above 44 Nm.
   - **Risked Failure:** OSF or TWF.
   - **Action:** Trigger an immediate physical inspection and reduce machine load.
