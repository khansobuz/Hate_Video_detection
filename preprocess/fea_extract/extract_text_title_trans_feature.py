import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

def extract_bert_features(dataset_dir, model_id, output_file, text_type='title_trans'):
    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            vid_file = os.path.join(dataset_dir, 'vids.csv')
            with open(vid_file, 'r') as f:
                self.vids = [line.strip() for line in f]

            # Load JSONL files
            ocr_file = os.path.join(dataset_dir, 'ocr.jsonl')
            trans_file = os.path.join(dataset_dir, 'speech.jsonl')
            title_file = os.path.join(dataset_dir, 'title.jsonl')
            self.ocr_df = pd.read_json(ocr_file, lines=True)
            self.trans_df = pd.read_json(trans_file, lines=True)
            self.title_df = pd.read_json(title_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]

            # Safe extraction: if vid not found, use empty string
            ocr_df_filt = self.ocr_df[self.ocr_df['vid'] == vid]
            trans_df_filt = self.trans_df[self.trans_df['vid'] == vid]
            title_df_filt = self.title_df[self.title_df['vid'] == vid]

            ocr = ocr_df_filt['ocr'].values[0] if len(ocr_df_filt) > 0 else ""
            trans = trans_df_filt['transcript'].values[0] if len(trans_df_filt) > 0 else ""
            title = title_df_filt['text'].values[0] if len(title_df_filt) > 0 else ""

            if text_type == 'title_trans':
                text = f'{title}\n{trans}'
            elif text_type == 'ocr_trans':
                text = f'{ocr}\n{trans}'
            else:
                raise ValueError("Invalid text_type. Choose 'title_trans' or 'ocr_trans'.")

            return vid, text

    def customed_collate_fn(batch):
        vids, texts = zip(*batch)
        inputs = processor(
            texts,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        return vids, inputs

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_id, device_map='cuda')
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
            inputs = inputs.to('cuda')
            pooler_output = model(**inputs)['last_hidden_state'][:, 0, :]
            pooler_output = pooler_output.detach().cpu()
            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]

    torch.save(save_dict, output_file)


# Extract features for all datasets
extract_bert_features(
    'data/MultiHateClip/zh',
    'local_models/bert-base-chinese',
    'data/MultiHateClip/zh/fea/fea_title_trans_bert-base-chinese.pt',
    text_type='title_trans'
)
extract_bert_features(
    'data/MultiHateClip/en',
    'local_models/bert-base-uncased',
    'data/MultiHateClip/en/fea/fea_title_trans_bert-base-uncased.pt',
    text_type='title_trans'
)
extract_bert_features(
    'data/HateMM',
    'local_models/bert-base-uncased',
    'data/HateMM/fea/fea_title_trans_bert-base-uncased.pt',
    text_type='title_trans'
)
