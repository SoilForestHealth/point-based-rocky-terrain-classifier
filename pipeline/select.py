import wandb

import pandas as pd
import json
import numpy as np

SWEEP_SUFFIX = "-tuning"

def select_models(models: list = None, sweep_ids: list = None):
    api = wandb.Api()

    best_model = None
    max_macro_avg_recall = -1

    for model, sweep_id in list(zip(models, sweep_ids)):
        sweep = api.sweep(f"gauravpendharkar/{model}{SWEEP_SUFFIX}/{sweep_id}")
        best_run = sweep.best_run()
        best_macro_avg_recall = json.loads(best_run.summary_metrics)['macro_recall_mean']
        print(f"Best run for model {model} is {best_run.id} with macro_recall_mean {best_macro_avg_recall}")
        
        model_artifacts = [artifact for artifact in best_run.logged_artifacts() if artifact.type == "model"]
        if model_artifacts:
            model_artifact = model_artifacts[0]
            model_dir = f"models/"
            model_artifact.download(model_dir)
            print(f"Model artifact for best run of {model} downloaded to {model_dir}")
        else:
            print(f"No model artifact found for the best run of {model}.")

        if best_macro_avg_recall > max_macro_avg_recall:
            max_macro_avg_recall = best_macro_avg_recall
            best_model = model

    print(f"Best model is {best_model} with macro_recall_mean {max_macro_avg_recall}")