# This is the file where we will extract the data from the csv file 
# and separate them into train / test split 

"""
IMPORTS 
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

"""
PATHS 
"""

RAW_PATH = "data/raw/ai_fingerprinting_dataset.csv"
OUT_DIR = "data/processed"

"""
WE EXTRACT THE DATA 
"""

# we get the data into a variable 
df = pd.read_csv(RAW_PATH)

# we get the columns that we need  
columns = [
        "prompt_id",
        "category",
        "prompt",
        "source_label",
        "generated_text"
    ]
# we make the new variable space with the only things we need 
df = df[columns].copy()

"""
TRAIN / TEST SPLIT 
"""

# to hold the seven labels together 
prompt_ids = df["prompt_id"].unique()

# here we have our seed to replicate the task 
train_prompt_ids, test_prompt_ids = train_test_split(
    prompt_ids,
    test_size=40,
    random_state=42,
    shuffle=True
)

train_df = df[df["prompt_id"].isin(train_prompt_ids)].copy()
test_df = df[df["prompt_id"].isin(test_prompt_ids)].copy()

train_df["split"] = "train"
test_df["split"] = "test"

full_df = pd.concat([train_df, test_df], ignore_index=True)

"""
SAVE FILES
"""

os.makedirs(OUT_DIR, exist_ok=True)

full_df.to_csv(f"{OUT_DIR}/full_dataset.csv", index=False)
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)


"""
CHECKS
"""

print("Processed files saved.")

print("\nFull dataset shape:", full_df.shape)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain label counts:")
print(train_df["source_label"].value_counts())

print("\nTest label counts:")
print(test_df["source_label"].value_counts())

overlap = set(train_df["prompt_id"]) & set(test_df["prompt_id"])
print("\nPrompt overlap between train and test:", len(overlap))
