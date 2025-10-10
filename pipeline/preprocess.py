# modules for data preprocessing
import pandas as pd
import numpy as np
import json

STRINGS = ["#DIV/0!", "#VALUE!"]

# inner join on a given key
def merge_dataframes(left, right, key):
    return pd.merge(left, right, on=list(key), how='inner')

# load and merge data 
# from multiple sources
def load_and_merge_data():

    with open("./data/metadata/data.json") as f:

        # metadata about datasets
        data_dir = json.load(f)

        # read relevant columns 
        # from each dataset
        relevant_data = []
        col_name_for_coordinate = ["COORDINATES", "coordinate", "COORDINATES", "COORDINATES"]
        col_name_for_depth = ["Depth (cm)", None, "Depth..cm.", "Depth..cm."]
        for dataset in data_dir.values():

            df = None
            relevant_cols = None if dataset["features"] == "all" else dataset["features"]

            # handle different file types
            if dataset["extension"] == "csv":
                df = pd.read_csv(dataset["file_path"], usecols=relevant_cols)
            elif dataset["extension"] == "xlsx":
                df = pd.read_excel(dataset["file_path"], usecols=relevant_cols)

            # use standard column names
            df = df.rename(columns={col_name_for_coordinate[dataset["id"]]: "ID",
                               col_name_for_depth[dataset["id"]]: "depth"})
            relevant_data.append(df)
        
        # merge all dataframes
        merged_df = relevant_data[0]
        for data in relevant_data[1:]:

            # find common keys to merge on
            key = set(merged_df.columns).intersection(["ID", "depth"])
            key = key.intersection(set(data.columns))
            merged_df = merge_dataframes(merged_df, data, key)
        
        # handle edge cases
        merged_df.replace(STRINGS, np.nan, inplace=True)
        merged_df = merged_df.rename(columns={"Organic Matter (%)": "organic_matter"})

        return merged_df

# add is_rocky_terrain column
def add_target_column(df):

    df["is_rocky_terrain"] = np.where(df["organic_matter"].isna(), "yes", "no")
    return df

# perform quality check
def perform_quality_check(df):

    mask_rocky_0_10 = (df["depth"] == "0-10") & (df["is_rocky_terrain"] == "yes")
    rocky_coords = df.loc[mask_rocky_0_10, "ID"]

    mask_10_20 = (df["depth"] == "10-20") & (df["ID"].isin(rocky_coords))
    df.loc[mask_10_20, "is_rocky_terrain"] = "yes"

    return df

# driver function
def preprocess_data():

    df = load_and_merge_data()
    df = add_target_column(df)
    df = perform_quality_check(df)
    return df