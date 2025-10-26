<div align="center">
<h1>Point-based Rocky Terrain Prediction</h1>
</div>

## Aim

To predict the presence of rocky terrain using topography and remote sensing data. This information includes:

1. Topography Features (aspect, elevation, flow length, plan curvature, profile curvature, slope, tan curvature, twi)

2. Remote Sensing Features (vegetation index, moisture index, bulk density, soil organic carbon, clay, sand)

<ins>**Note**:</ins> Every coordinate is considered in isolation and hence this model follows a point-based approach.

## Environment Setup

- Clone this repository.

```bash
git clone https://github.com/SoilForestHealth/rocky-terrain-classifier.git
```

- Install all packages in the ```requirements.txt``` file.

```bash
pip install -r requirements.txt
```

- The following directory structure is required for the code in this repository to work properly:
```bash
├── data
│   ├── LSF_Grid_Soil_Data_2025_Summer.xlsx
│   ├── LSF_Topography_Covariates_2025_Summer.csv
│   ├── metadata
│   │   ├── data.json
│   │   ├── system.json
│   │   └── tune.json
│   ├── raw_field_summer_2025_covariates_combined.csv
│   └── s2_cloudless_covariates_field_summer_2025_combined.csv
├── main.py
├── pipeline
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocess.py
│   └── select.py
└── requirements.txt

4 directories, 13 files
```

- To execute the code in this repository, run `main.py` file. Ensure you are in the root directory of the repository.

```bash
python3 main.py
```

- Feel free to raise an issue if there are any problems with the repository!

## Results

To evaluate the models for the imbalanced binary classification problem, we use `macro_avg_recall` and `f2_score`.

<div align="center">

$f2\\\_score = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$

$macro\\\_avg\\\_recall = \frac{1}{2} \sum_{i=1}^{2} R_i$


| Model    | macro_avg_recall    |  f2_score  |  wandb sweeps  |
|:--------------|:--------------:|:--------------:|:--------------:|
| Logistic Regression | 0.557 | 0.557 | [wandb](https://wandb.ai/gauravpendharkar/logistic-regression-tuning) |
| Decision Tree | 0.654 | 0.658 | [wandb](https://wandb.ai/gauravpendharkar/decision-tree-tuning) |
| **Random Forest** | **0.723**| **0.731** | [wandb](https://wandb.ai/gauravpendharkar/random-forest-tuning) |
| Extra Trees | 0.696 | 0.700 | [wandb](https://wandb.ai/gauravpendharkar/extra-trees-tuning) |
| Gradient Boosting Trees | 0.724 | 0.724 | [wandb](https://wandb.ai/gauravpendharkar/gradient-boosting-trees-tuning) |

</div>

The overall model comparison dashboard is [here](https://wandb.ai/gauravpendharkar/model-comparison).
