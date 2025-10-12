# modules for data preprocessing
import pandas as pd
import numpy as np
import json

# modules for model building
from sklearn.model_selection import train_test_split
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

# modules for abstract class
from abc import ABC, abstractmethod

# modules for hyperparameter tuning
import wandb

class Model(ABC):
    def __init__(self, model_name: str, model: any):
        self.model_name = model_name
        self.model = model
        self.selected_features = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    @abstractmethod
    def select_features(self, X_train: any, y_train: any):
        pass

    def train_sweep(self, config=None):
        with wandb.init(config=config, project=self.project_name) as run:
            cfg = run.config

            params = dict(cfg)
            if "class_weight" in cfg:
                cw_str = cfg.class_weight
                class_weight = {i: float(v) for i, v in enumerate(cw_str.split("_"))}

                params["class_weight"] = class_weight

            print(f"Training with config: {params}")

            # Set params for the model
            self.model.set_params(**params)

            # Prepare CV
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Store metrics across folds
            precisions, recalls, f2_scores, aucs, pr_aucs = [], [], [], [], []

            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                # Train and predict
                self.model.fit(X_tr, y_tr)
                y_proba = self.model.predict_proba(X_val)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                # Compute metrics
                precisions.append(precision_score(y_val, y_pred, average="macro", zero_division=0))
                recalls.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
                f2_scores.append(fbeta_score(y_val, y_pred, average="macro", beta=2, zero_division=0))
                aucs.append(roc_auc_score(y_val, y_proba))
                pr_aucs.append(average_precision_score(y_val, y_proba))

            # Compute mean & std for all metrics
            results = {
                "macro_precision_mean": np.mean(precisions),
                "macro_recall_mean": np.mean(recalls),
                "macro_f2_mean": np.mean(f2_scores),
                "roc_auc_mean": np.mean(aucs),
                "pr_auc_mean": np.mean(pr_aucs),
                "macro_precision_std": np.std(precisions),
                "macro_recall_std": np.std(recalls),
                "macro_f2_std": np.std(f2_scores),
                "roc_auc_std": np.std(aucs),
                "pr_auc_std": np.std(pr_aucs),
            }

            run.log(results)

    def tune(self, X_train, X_test, y_train, y_test, sweep_config: dict, project_name):
        
        # Save references for sweep
        self.X_train = X_train[self.selected_features] if self.selected_features is not None else X_train
        self.X_test = X_test[self.selected_features] if self.selected_features is not None else X_test
        self.y_train = y_train
        self.y_test = y_test
        self.project_name = project_name

        # Initialize and run sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=self.train_sweep, count=sweep_config.get("count", 50))
