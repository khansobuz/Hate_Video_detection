import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def extract_bert_features(dataset_dir, model_id, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Dataset
    # -------------------------
    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            vid_file = os.path.join(dataset_dir, "vids.csv")
            with open(vid_file, "r", encoding="utf-8") as f:
                self.vids = [line.strip() for line in f]

            text_file = os.path.join(dataset_dir, "caption.jsonl")
            self.text_df = pd.read_json(text_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]
            row = self.text_df[self.text_df["vid"] == vid]

            # if caption missing → use empty string
            if len(row) == 0:
                text = ""
            else:
                text = row["text"].values[0]

            return vid, text

    # -------------------------
    # Collate
    # -------------------------
    def collate_fn(batch):
        vids, texts = zip(*batch)
        inputs = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return vids, inputs

    # -------------------------
    # Model & tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # -------------------------
    # DataLoader
    # -------------------------
    dataloader = DataLoader(
        MyDataset(dataset_dir),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    save_dict = {}

    # -------------------------
    # Feature extraction
    # -------------------------
    with torch.no_grad():
        for vids, inputs in tqdm(dataloader, desc=f"Extracting {dataset_dir}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # CLS token
            features = outputs.last_hidden_state[:, 0, :].cpu()

            for i, vid in enumerate(vids):
                save_dict[vid] = features[i]

    # -------------------------
    # SAVE (IMPORTANT FIX)
    # -------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(save_dict, output_file)
    print(f"✅ Saved: {output_file}")


# =========================
# RUN FOR ALL DATASETS
# =========================
extract_bert_features(
    "data/MultiHateClip/zh",
    "local_models/bert-base-chinese",
    "data/MultiHateClip/zh/fea/fea_caption_bert-base-chinese.pt",
)

extract_bert_features(
    "data/MultiHateClip/en",
    "local_models/bert-base-uncased",
    "data/MultiHateClip/en/fea/fea_caption_bert-base-uncased.pt",
)

extract_bert_features(
    "data/HateMM",
    "local_models/bert-base-uncased",
    "data/HateMM/fea/fea_caption_bert-base-uncased.pt",
)
