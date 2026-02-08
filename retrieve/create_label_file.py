import pandas as pd
import json
from pathlib import Path

def create_label_jsonl(tsv_file: Path, output_file: Path):
    # Read TSV file
    df = pd.read_csv(tsv_file, sep="\t")  # TSV uses tab separator

    # Map labels to numeric
    # Assuming 'Normal' -> 0, everything else (Offensive, Hateful, etc.) -> 1
    def map_label(label):
        if str(label).lower() == "normal":
            return 0
        else:
            return 1

    # The column name with labels may differ. Often it's 'Majority_Voting' or 'Label'
    # Adjust if needed
    if "Majority_Voting" in df.columns:
        label_col = "Majority_Voting"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise ValueError("Cannot find label column in TSV file.")

    df["numeric_label"] = df[label_col].apply(map_label)

    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_obj = {"vid": row["Video_ID"], "label": row["numeric_label"]}
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"Saved {output_file}")

# Paths
en_tsv = Path("C:/Users/khanm/Desktop/lab_project/HVD/data/MultiHateClip/en/annotation/all.tsv")
zh_tsv = Path("C:/Users/khanm/Desktop/lab_project/HVD/data/MultiHateClip/zh/annotation/all.tsv")

en_output = Path("C:/Users/khanm/Desktop/lab_project/HVD/data/MultiHateClip/en/label.jsonl")
zh_output = Path("C:/Users/khanm/Desktop/lab_project/HVD/data/MultiHateClip/zh/label.jsonl")

# Create JSONL files
create_label_jsonl(en_tsv, en_output)
create_label_jsonl(zh_tsv, zh_output)
