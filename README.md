# League of Legends Match Outcome Predictor (PyTorch & Scikit-Learn)

## Project Overview
An end-to-end machine learning classification pipeline built in **PyTorch** and **Scikit-Learn** to predict League of Legends match outcomes (`win` vs. `loss`) from in-game telemetry and performance metrics.

## Tech Stack
* **Language:** Python
* **Deep Learning & ML:** PyTorch (`torch.nn`, `torch.optim`), Scikit-Learn (`StandardScaler`, `train_test_split`, `classification_report`, `confusion_matrix`, `roc_auc_score`)
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib

## Pipeline Architecture (8-Step Workflow)
1. **Data Loading & Preprocessing:** Ingested match telemetry data (`league_of_legends_data_large.csv`), separated features and target labels, applied an 80/20 train-test split, and standardized features via `StandardScaler`.
2. **Model Implementation:** Constructed a custom PyTorch `LogisticRegressionModel` utilizing `nn.Linear` and `torch.sigmoid` activation.
3. **Model Training:** Trained over 1,000 epochs using Binary Cross-Entropy Loss (`nn.BCELoss`) and Stochastic Gradient Descent (`optim.SGD`).
4. **L2 Regularization Optimization:** Integrated weight decay (`weight_decay=0.01`) to penalize overly complex weights and reduce overfitting.
5. **Model Evaluation & Visualization:** Generated Classification Reports, Confusion Matrices, and Receiver Operating Characteristic (ROC-AUC) curves on unseen test data.
6. **Model Persistence:** Saved and verified model state dictionaries (`lol_model_weights.pth`) using `torch.save` and `load_state_dict`.
7. **Hyperparameter Tuning:** Evaluated learning rates across `[0.01, 0.05, 0.1]` by reinitializing model weights per configuration, identifying `lr = 0.05` as the optimal rate (52.00% test accuracy).
8. **Feature Importance Extraction:** Analyzed learned linear layer weights to rank in-game metrics by predictive impact.

## Key Feature Importance Findings
| Rank | Match Feature | Learned Weight | Impact Direction |
| :--- | :--- | :--- | :--- |
| **1** | `gold_earned` | `+0.1685` | Strongest positive predictor of match victory |
| **2** | `kills` | `+0.1261` | Primary offensive driver of match victory |
| **3** | `wards_placed` | `+0.1021` | Strong map vision & objective control correlation |
| **4** | `deaths` | `-0.0045` | Negative impact on win probability |
| **5** | `cs` | `-0.0071` | Secondary impact (collinear with `gold_earned`) |
| **6** | `assists` | `-0.0130` | Secondary impact (collinear with `kills`) |
| **7** | `damage_dealt` | `-0.0176` | Secondary impact (collinear with `kills` & `gold_earned`) |
| **8** | `wards_killed` | `-0.0326` | Minor defensive vision metric |
