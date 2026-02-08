import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def extract_bert_features(dataset_dir, model_id, output_file):
    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            vid_file = os.path.join(dataset_dir, "vids.csv")
            with open(vid_file, "r") as f:
                self.vids = [line.strip() for line in f]
            text_file = os.path.join(dataset_dir, "title.jsonl")
            self.text_df = pd.read_json(text_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]
            filtered = self.text_df[self.text_df["vid"] == vid]
            if len(filtered) == 0:
                # Vid missing in text file, use empty string
                text = ""
                print(f"Warning: missing vid {vid}, using empty text")
            else:
                text = filtered["text"].values[0]
            return vid, text

    def customed_collate_fn(batch):
        vids, texts = zip(*batch)
        inputs = processor(
            texts, padding="max_length", truncation=True, return_tensors="pt", max_length=512
        )
        return vids, inputs

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_id, device_map="cuda")
    processor = AutoTokenizer.from_pretrained(model_id)

    save_dict = {}

    dataloader = DataLoader(
        MyDataset(dataset_dir),
        batch_size=1,
        collate_fn=customed_collate_fn,
        num_workers=0,
        shuffle=False
    )

    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            vids, inputs = batch
            inputs = inputs.to("cuda")
            pooler_output = model(**inputs)["last_hidden_state"][:, 0, :]
            pooler_output = pooler_output.detach().cpu()
            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]

    # Save features
    torch.save(save_dict, output_file)
    print(f"Saved BERT features to {output_file}")


# === Extract English features ===
extract_bert_features(
    "data/MultiHateClip/en",
    "local_models/bert-base-uncased",  # your local English BERT path
    "data/MultiHateClip/en/fea/fea_title_bert-base-uncased.pt",
)

# === Extract Chinese features ===
extract_bert_features(
    "data/MultiHateClip/zh",
    "local_models/bert-base-chinese",  # your local Chinese BERT path
    "data/MultiHateClip/zh/fea/fea_title_bert-base-chinese.pt",
)
