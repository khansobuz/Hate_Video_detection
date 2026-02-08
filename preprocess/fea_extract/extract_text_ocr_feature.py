import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


def extract_bert_features(dataset_dir, model_id, output_file):

    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            # load video ids
            vid_file = os.path.join(dataset_dir, 'vids.csv')
            with open(vid_file, 'r', encoding='utf-8') as f:
                self.vids = [line.strip() for line in f]

            # load OCR text
            text_file = os.path.join(dataset_dir, 'ocr.jsonl')
            self.text_df = pd.read_json(text_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]

            # SAFE OCR lookup (fixes your error)
            rows = self.text_df[self.text_df['vid'] == vid]
            if len(rows) > 0:
                text = rows['ocr'].values[0]
            else:
                text = ""  # empty text if OCR missing

            return vid, text

    def customed_collate_fn(batch):
        vids, texts = zip(*batch)
        inputs = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return vids, inputs

    # Load model & tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    save_dict = {}

    dataloader = DataLoader(
        MyDataset(dataset_dir),
        batch_size=1,
        collate_fn=customed_collate_fn,
        shuffle=False,
        num_workers=0
    )

    model.eval()
    for vids, inputs in tqdm(dataloader):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # CLS token embedding
            outputs = model(**inputs)
            cls_feature = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            cls_feature = cls_feature.cpu()

            for i, vid in enumerate(vids):
                save_dict[vid] = cls_feature[i]

    # save features
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(save_dict, output_file)
    print(f"Saved features to {output_file}")


# ======================
# Run feature extraction
# ======================
extract_bert_features(
    'data/MultiHateClip/zh',
    'local_models/bert-base-chinese',
    'data/MultiHateClip/zh/fea/fea_ocr_bert-base-chinese.pt'
)

extract_bert_features(
    'data/MultiHateClip/en',
    'local_models/bert-base-uncased',
    'data/MultiHateClip/en/fea/fea_ocr_bert-base-uncased.pt'
)

extract_bert_features(
    'data/HateMM',
    'local_models/bert-base-uncased',
    'data/HateMM/fea/fea_ocr_bert-base-uncased.pt'
)
