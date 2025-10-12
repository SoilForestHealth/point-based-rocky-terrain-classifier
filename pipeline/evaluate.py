# modules for data preprocessing
import pandas as pd
import numpy as np
import json

# modules for model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# modules for cross-validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score
)

# modules for hyperparameter tuning
import wandb

class Evaluator:
    def __init__(self, model: any, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        self.y_pred = None
        self.y_prob_1d = None
        self.y_prob_2d = None
        self.metrics = None
        self.feature_importances_df = None

    def compute_metrics(self):

        self.y_pred = self.model.predict(self.X_test)

        self.y_prob_1d = self.model.predict_proba(self.X_test)[:, 1]
        self.y_prob_2d = self.model.predict_proba(self.X_test)

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            importances = np.zeros(self.X_test.shape[1])
            
        self.feature_importances_df = (
            pd.DataFrame({
                'feature': self.X_test.columns,
                'importance': importances
            })
            .sort_values(by='importance', ascending=False)
            .reset_index(drop=True)
        )

        feature_table = wandb.Table(dataframe=self.feature_importances_df)

        self.metrics = {
            "test/precision": precision_score(self.y_test, self.y_pred, average='macro'),
            "test/recall": recall_score(self.y_test, self.y_pred, average='macro'),
            "test/f1_score": f1_score(self.y_test, self.y_pred, average='macro'),
            "test/f2_score": fbeta_score(self.y_test, self.y_pred, beta=2, average='macro'),
            "test/roc_auc": roc_auc_score(self.y_test, self.y_prob_1d),
            "test/average_precision": average_precision_score(self.y_test, self.y_prob_1d),
             "test/feature_importances_table": feature_table
        }
    
    def log_metrics(self):

        with wandb.init(project="model-comparison", reinit=True) as run:

            y_true_clean = np.asarray(self.y_test).flatten().astype(int)
            y_pred_clean = np.asarray(self.y_pred).flatten().astype(int)
            y_prob_2d_clean = np.asarray(self.y_prob_2d)

            plots = {
                "roc_curve": wandb.plot.roc_curve(
                    y_true_clean, 
                    y_prob_2d_clean,
                    classes_to_plot=[1]
                ),
                "pr_curve": wandb.plot.pr_curve(
                    y_true_clean, 
                    y_prob_2d_clean,
                    classes_to_plot=[1]
                ),
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true_clean, 
                    preds=y_pred_clean,
                    class_names=["Non-rocky", "Rocky"]
                )
            }
        
            all_metrics = self.metrics | plots 
            run.log(all_metrics)