"""
MoRE (Modality Retrieval Expert) Model Training
Consolidated single-file implementation for HateMM dataset
"""

import sys
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_worker_seed(worker_id, seed):
    """Set seed for dataloader workers"""
    import random
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def worker_init(worker_id, seed):
    """Initialize worker with seed"""
    set_worker_seed(worker_id, seed)


# ============================================================================
# BASE DATASET CLASS
# ============================================================================

class HateMM_Dataset(Dataset):
    """Base dataset class for HateMM"""
    def __init__(self):
        super().__init__()
    def _get_data(self, fold: int, split: str, task: str):
        """Load data for specific fold and split"""
        data_path = Path('data/HateMM')
        data_file = data_path / f'{split}.csv'
        label_file = data_path / 'label.jsonl'
        
        if not data_file.exists():
            raise FileNotFoundError(f'Data file not found: {data_file}')
        if not label_file.exists():
            raise FileNotFoundError(f'Label file not found: {label_file}')
        
        # Read video IDs from CSV
        df = pd.read_csv(data_file, header=None, names=['Video_ID'])
        
        # Read labels from JSONL
        labels_df = pd.read_json(label_file, lines=True)
        
        # Merge to get labels for each video
        df = df.merge(labels_df, left_on='Video_ID', right_on='vid', how='left')
        
        # Drop the duplicate 'vid' column
        if 'vid' in df.columns:
            df = df.drop(columns=['vid'])
        
        print(f"Loaded {split}.csv: {len(df)} rows")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        print(f"First few rows:\n{df.head()}")
        
        # Check for missing labels
        missing_labels = df['label'].isna().sum()
        if missing_labels > 0:
            print(f"WARNING: {missing_labels} videos have no labels in label.jsonl")
            df = df.dropna(subset=['label'])
            df = df.reset_index(drop=True)
            print(f"After removing missing labels: {len(df)} rows")
        
        return df


# ============================================================================
# DATASET AND COLLATOR
# ============================================================================

class HateMM_MoRE_Dataset(HateMM_Dataset):
    """MoRE dataset for HateMM with retrieval features"""
    def __init__(self, fold: int, split: str, task: str, ablation='No', num_pos=30, num_neg=30, **kwargs):
        super(HateMM_MoRE_Dataset, self).__init__()
        fea_path = Path('data/HateMM/fea')
        sim_path = Path('data/HateMM/retrieval')
        self.data = self._get_data(fold, split, task)
        self.mfcc_fea = torch.load(fea_path / 'fea_audio_mfcc.pt', weights_only=True)
        self.text_fea = torch.load(fea_path / 'fea_transcript_bert-base-uncased.pt', weights_only=True)
        self.frame_fea = torch.load(fea_path / 'fea_frames_16_google-vit-base-16-224.pt', weights_only=True)
        self.sim_all_sim = pd.read_json(sim_path / 'all_modal.jsonl', lines=True)
        self.ablation = ablation
        self.num_pos = num_pos
        self.num_neg = num_neg
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # Handle different possible column names for label
        if 'label' in item:
            label = item['label']
        elif 'Label' in item:
            label = item['Label']
        elif 'target' in item:
            label = item['target']
        elif len(item) == 2:  # If only 2 columns, assume [Video_ID, label]
            label = item.iloc[1]
        else:
            raise KeyError(f"Could not find label column. Available columns: {list(item.index)}")
        # Handle NaN or invalid labels
        if pd.isna(label):
            raise ValueError(f"Label is NaN for index {idx}, Video_ID: {item.iloc[0]}")
        label = int(float(label))  # Convert to float first to handle string numbers, then to int
        # Get Video_ID
        if 'Video_ID' in item:
            vid = item['Video_ID']
        elif 'video_id' in item:
            vid = item['video_id']
        elif 'vid' in item:
            vid = item['vid']
        elif len(item) >= 1:
            vid = item.iloc[0]
        else:
            raise KeyError(f"Could not find Video_ID column. Available columns: {list(item.index)}")
        text_fea = self.text_fea[vid]
        vision_fea = self.frame_fea[vid]
        audio_fea = self.mfcc_fea[vid]
        num_pos = self.num_pos
        num_neg = self.num_neg
        sim_data = self.sim_all_sim[self.sim_all_sim['vid'] == vid].iloc[0]['similarities']
        all_sim_pos_vids = sim_data[0]['vid'][:num_pos]
        all_sim_neg_vids = sim_data[1]['vid'][:num_neg]
        text_sim_pos_fea = torch.stack([self.text_fea[key] for key in all_sim_pos_vids])
        text_sim_neg_fea = torch.stack([self.text_fea[key] for key in all_sim_neg_vids])
        vision_sim_pos_fea = torch.stack([self.frame_fea[key] for key in all_sim_pos_vids])
        vision_sim_neg_fea = torch.stack([self.frame_fea[key] for key in all_sim_neg_vids])
        audio_sim_pos_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_pos_vids])
        audio_sim_neg_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_neg_vids])
        return {
            'vid': vid, 'label': torch.tensor(label, dtype=torch.long), 'text_fea': text_fea, 'vision_fea': vision_fea,
            'audio_fea': audio_fea, 'text_sim_pos_fea': text_sim_pos_fea, 'text_sim_neg_fea': text_sim_neg_fea,
            'vision_sim_pos_fea': vision_sim_pos_fea, 'vision_sim_neg_fea': vision_sim_neg_fea,
            'audio_sim_pos_fea': audio_sim_pos_fea, 'audio_sim_neg_fea': audio_sim_neg_fea,
        }


class HateMM_MoRE_Collator:
    """Collator for batching HateMM data"""
    def __init__(self, **kwargs):
        pass
    def __call__(self, batch):
        return {
            'vids': [item['vid'] for item in batch],
            'labels': torch.stack([item['label'] for item in batch]),
            'text_fea': torch.stack([item['text_fea'] for item in batch]),
            'vision_fea': torch.stack([item['vision_fea'] for item in batch]),
            'audio_fea': torch.stack([item['audio_fea'] for item in batch]),
            'text_sim_pos_fea': torch.stack([item['text_sim_pos_fea'] for item in batch]),
            'text_sim_neg_fea': torch.stack([item['text_sim_neg_fea'] for item in batch]),
            'vision_sim_pos_fea': torch.stack([item['vision_sim_pos_fea'] for item in batch]),
            'vision_sim_neg_fea': torch.stack([item['vision_sim_neg_fea'] for item in batch]),
            'audio_sim_pos_fea': torch.stack([item['audio_sim_pos_fea'] for item in batch]),
            'audio_sim_neg_fea': torch.stack([item['audio_sim_neg_fea'] for item in batch]),
        }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequences"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class AttentivePooling(nn.Module):
    """Attentive pooling layer"""
    
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
    
    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return (weights * x).sum(dim=1)


def check_shape(tensor, expected_shape, name="tensor"):
    """Utility to check tensor shape"""
    if tensor.shape != expected_shape:
        logger.warning(f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}")


class ModalityExpert(nn.Module):
    """Expert module for each modality"""
    def __init__(self, hid_dim, dropout, num_head, alpha, ablation):
        super().__init__()
        self.ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
        self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.alpha = alpha
        self.ablation = ablation
    def forward(self, query, pos, neg):
        query = self.ffn(query)
        pos = self.ffn(pos)
        neg = self.ffn(neg)
        # Ensure query has 3 dimensions for batch_first=True
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, D]
        pos_attn, _ = self.pos_attn(query, pos, pos)
        neg_attn, _ = self.neg_attn(query, neg, neg)
        ret = self.alpha * pos_attn + (1 - self.alpha) * neg_attn + query
        return ret


# ============================================================================
# MAIN MODEL
# ============================================================================

class MoRE(nn.Module):
    """Modality Retrieval Expert (MoRE) Model"""
    
    name = 'MoRE'
    
    def __init__(self, text_encoder, fea_dim=768, dropout=0.2, num_head=8, 
                 alpha=0.5, delta=0.25, num_epoch=20, ablation='No', 
                 loss='No', **kwargs):
        super().__init__()

        self.bert = AutoModel.from_pretrained(text_encoder).requires_grad_(False)
        if hasattr(self.bert, 'text_model'):
            self.bert = self.bert.text_model
        
        self.text_ffn = nn.LazyLinear(fea_dim)
        self.vision_ffn = nn.LazyLinear(fea_dim)
        self.audio_ffn = nn.LazyLinear(fea_dim)
        
        self.text_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
        self.vision_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
        self.audio_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
        
        self.positional_encoding = LearnablePositionalEncoding(768, 16)
        
        self.text_pre_router = nn.LazyLinear(fea_dim)
        self.vision_pre_router = nn.LazyLinear(fea_dim)
        self.audio_pre_router = nn.LazyLinear(fea_dim)
        
        self.router = nn.Sequential(
            nn.LazyLinear(fea_dim), 
            nn.ReLU(), 
            nn.LazyLinear(3), 
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
            nn.LazyLinear(200), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(200, 2)
        )

        self.text_preditor = nn.Sequential(
            nn.LazyLinear(fea_dim), 
            nn.ReLU(), 
            nn.LazyLinear(2)
        )
        self.vision_preditor = nn.Sequential(
            nn.LazyLinear(fea_dim), 
            nn.ReLU(), 
            nn.LazyLinear(2)
        )
        self.audio_preditor = nn.Sequential(
            nn.LazyLinear(fea_dim), 
            nn.ReLU(), 
            nn.LazyLinear(2)
        )
        
        self.text_pooler = AttentivePooling(fea_dim)
        self.vision_pooler = AttentivePooling(fea_dim)
        
        self.delta = delta
        self.total_epoch = num_epoch
        self.ablation = ablation
        self.loss = loss

    def forward(self, **inputs):
        text_fea = inputs['text_fea']
        audio_fea = inputs['audio_fea']
        vision_fea = inputs['vision_fea']
        text_sim_pos_fea = inputs['text_sim_pos_fea']
        text_sim_neg_fea = inputs['text_sim_neg_fea']
        frame_sim_pos_fea = inputs['vision_sim_pos_fea']
        frame_sim_neg_fea = inputs['vision_sim_neg_fea']
        mfcc_sim_pos_fea = inputs['audio_sim_pos_fea']
        mfcc_sim_neg_fea = inputs['audio_sim_neg_fea']
        # Ensure text_fea and audio_fea have 3 dimensions
        if text_fea.dim() == 2:
            text_fea = text_fea.unsqueeze(1)  # [B, 1, D]
        if audio_fea.dim() == 2:
            audio_fea = audio_fea.unsqueeze(1)  # [B, 1, D]
        # Apply positional encoding to vision features
        vision_fea = self.positional_encoding(vision_fea)
        # For similarity features, they come as [B, N, L, D]
        # Reshape to [B*N, L, D] for positional encoding, then back to [B, N*L, D]
        B = frame_sim_pos_fea.shape[0]
        N_pos = frame_sim_pos_fea.shape[1]
        N_neg = frame_sim_neg_fea.shape[1]
        L = frame_sim_pos_fea.shape[2]
        D = frame_sim_pos_fea.shape[3]
        frame_sim_pos_fea_flat = frame_sim_pos_fea.reshape(B*N_pos, L, D)
        frame_sim_pos_fea_encoded = self.positional_encoding(frame_sim_pos_fea_flat)
        frame_sim_pos_fea = frame_sim_pos_fea_encoded.reshape(B, N_pos*L, D)
        frame_sim_neg_fea_flat = frame_sim_neg_fea.reshape(B*N_neg, L, D)
        frame_sim_neg_fea_encoded = self.positional_encoding(frame_sim_neg_fea_flat)
        frame_sim_neg_fea = frame_sim_neg_fea_encoded.reshape(B, N_neg*L, D)
        # Apply modality experts
        text_fea_aug = self.text_expert(text_fea, text_sim_pos_fea, text_sim_neg_fea)
        vision_fea_aug = self.vision_expert(vision_fea, frame_sim_pos_fea, frame_sim_neg_fea)
        audio_fea_aug = self.audio_expert(audio_fea, mfcc_sim_pos_fea, mfcc_sim_neg_fea)
        # Pooling
        vision_fea = vision_fea.mean(dim=1, keepdim=True)
        router_fea = torch.cat([text_fea, vision_fea, audio_fea], dim=-1)
        weight = self.router(router_fea).squeeze(1)
        text_fea_aug = text_fea_aug.mean(dim=1)
        vision_fea_aug = self.vision_pooler(vision_fea_aug)
        audio_fea_aug = audio_fea_aug.mean(dim=1)
        # Individual predictions
        text_pred = self.text_preditor(text_fea_aug)
        vision_pred = self.vision_preditor(vision_fea_aug)
        audio_pred = self.audio_preditor(audio_fea_aug)
        # Fusion
        if self.ablation == 'w/o-router':
            fea = (text_fea_aug + vision_fea_aug + audio_fea_aug) / 3
        else:
            fea = (text_fea_aug * weight[:, 0].unsqueeze(1) + 
                   vision_fea_aug * weight[:, 1].unsqueeze(1) + 
                   audio_fea_aug * weight[:, 2].unsqueeze(1))
        output = self.classifier(fea)
        return {
            'pred': output,
            'text_pred': text_pred,
            'vision_pred': vision_pred,
            'audio_pred': audio_pred,
            'weight': weight,
        }
    
    def calculate_loss(self, **inputs):
        delta = self.delta
        total_epoch = self.total_epoch
        
        pred = inputs['pred']
        label = inputs['label']
        text_pred = inputs['text_pred']
        vision_pred = inputs['vision_pred']
        audio_pred = inputs['audio_pred']
        cur_epoch = inputs['epoch']
        
        f_epo = (float(cur_epoch) / float(total_epoch)) ** 2

        l_mix = F.cross_entropy(pred, label)
        
        text_loss = F.cross_entropy(text_pred, label) if text_pred is not None else 0.0
        vision_loss = F.cross_entropy(vision_pred, label) if vision_pred is not None else 0.0
        audio_loss = F.cross_entropy(audio_pred, label) if audio_pred is not None else 0.0

        l_exp = (text_loss + vision_loss + audio_loss) / 3
        l_join = min(1 - f_epo, delta) * l_exp + max(f_epo, 1 - delta) * l_mix
        
        return l_join, l_mix


# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

class BinaryClassificationMetric:
    """Metrics for binary classification"""
    
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
    
    def update(self, pred, label):
        self.preds.append(pred.cpu())
        self.labels.append(label.cpu())
    
    def compute(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        metrics = {
            'acc': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro'),
            'macro_prec': precision_score(labels, preds, average='macro', zero_division=0),
            'macro_rec': recall_score(labels, preds, average='macro', zero_division=0),
            'a_f1': f1_score(labels, preds, pos_label=0, zero_division=0),
            'a_prec': precision_score(labels, preds, pos_label=0, zero_division=0),
            'a_rec': recall_score(labels, preds, pos_label=0, zero_division=0),
            'b_f1': f1_score(labels, preds, pos_label=1, zero_division=0),
            'b_prec': precision_score(labels, preds, pos_label=1, zero_division=0),
            'b_rec': recall_score(labels, preds, pos_label=1, zero_division=0),
        }
        
        self.reset()
        return metrics


class TernaryClassificationMetric:
    """Metrics for ternary classification"""
    
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
    
    def update(self, pred, label):
        self.preds.append(pred.cpu())
        self.labels.append(label.cpu())
    
    def compute(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        metrics = {
            'acc': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro'),
            'macro_prec': precision_score(labels, preds, average='macro', zero_division=0),
            'macro_rec': recall_score(labels, preds, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for i, label_name in enumerate(['a', 'b', 'c']):
            metrics[f'{label_name}_f1'] = f1_score(labels, preds, labels=[i], average=None, zero_division=0)[0]
            metrics[f'{label_name}_prec'] = precision_score(labels, preds, labels=[i], average=None, zero_division=0)[0]
            metrics[f'{label_name}_rec'] = recall_score(labels, preds, labels=[i], average=None, zero_division=0)[0]
        
        self.reset()
        return metrics


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=5, path='best_model.pth', delta=0):
        self.patience = patience
        self.path = path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss, model):
        score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ============================================================================
# OPTIMIZER AND SCHEDULER
# ============================================================================

def get_optimizer(model, name='AdamW', lr=5e-4, weight_decay=5e-5, **kwargs):
    """Get optimizer"""
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {name} not supported')


def get_scheduler(optimizer, name='DummyLR', steps_per_epoch=100, **kwargs):
    """Get learning rate scheduler"""
    if name == 'DummyLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f'Scheduler {name} not supported')


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Main training class"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = cfg.task
        
        if cfg.task == 'binary':
            self.evaluator = BinaryClassificationMetric(self.device)
        elif cfg.task == 'ternary':
            self.evaluator = TernaryClassificationMetric(self.device)
        else:
            raise ValueError('task not supported')
        
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        if cfg.type == 'default':
            self.dataset_range = ['default']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = HateMM_MoRE_Collator(**cfg.data)

    def _reset(self, cfg, fold, type):
        """Reset model and data for new fold"""
        train_dataset = HateMM_MoRE_Dataset(task=cfg.task, fold=fold, split='train', **cfg.data)
        test_dataset = HateMM_MoRE_Dataset(task=cfg.task, fold=fold, split='test', **cfg.data)
        if cfg.task == 'binary':
            valid_dataset = HateMM_MoRE_Dataset(task=cfg.task, fold=fold, split='valid', **cfg.data)
        num_workers = min(16, max(1, cfg.batch_size // 4))
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
            num_workers=num_workers, shuffle=True, generator=self.generator, worker_init_fn=partial(worker_init, seed=cfg.seed))
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
            num_workers=num_workers, shuffle=False, generator=self.generator, worker_init_fn=partial(worker_init, seed=cfg.seed))
        if cfg.task == 'binary':
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
                num_workers=num_workers, shuffle=False, generator=self.generator, worker_init_fn=partial(worker_init, seed=cfg.seed))
        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = MoRE(**dict(cfg.para))
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path / 'best_model.pth')

    def run(self):
        """Main training loop"""
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        a_f1_list, a_prec_list, a_rec_list = [], [], []
        b_f1_list, b_prec_list, b_rec_list = [], [], []
        c_f1_list, c_prec_list, c_rec_list = [], [], []
        
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train(epoch=epoch)
                
                if self.task == 'binary':
                    self._valid(split='valid', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                        break
                    self._valid(split='test', epoch=epoch)
                elif self.task == 'ternary':
                    self._valid(split='test', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.RED}Early stopping at epoch {epoch}")
                        break
            
            logger.info(f'{Fore.RED}Best results for fold {fold}:')
            self.model.load_state_dict(
                torch.load(self.save_path / 'best_model.pth', weights_only=False)
            )
            best_metrics = self._valid(split='test', epoch=epoch, final=True)
            
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['macro_f1'])
            prec_list.append(best_metrics['macro_prec'])
            rec_list.append(best_metrics['macro_rec'])
            a_f1_list.append(best_metrics['a_f1'])
            a_prec_list.append(best_metrics['a_prec'])
            a_rec_list.append(best_metrics['a_rec'])
            b_f1_list.append(best_metrics['b_f1'])
            b_prec_list.append(best_metrics['b_prec'])
            b_rec_list.append(best_metrics['b_rec'])
            
            if self.task == 'ternary':
                c_f1_list.append(best_metrics['c_f1'])
                c_prec_list.append(best_metrics['c_prec'])
                c_rec_list.append(best_metrics['c_rec'])
        
        # Final results
        logger.info(f'Average Acc: {np.mean(acc_list):.5f}, F1: {np.mean(f1_list):.5f}, '
                   f'Precision: {np.mean(prec_list):.5f}, Recall: {np.mean(rec_list):.5f}')
        logger.info(f'Class A - F1: {np.mean(a_f1_list):.5f}, '
                   f'Precision: {np.mean(a_prec_list):.5f}, Recall: {np.mean(a_rec_list):.5f}')
        logger.info(f'Class B - F1: {np.mean(b_f1_list):.5f}, '
                   f'Precision: {np.mean(b_prec_list):.5f}, Recall: {np.mean(b_rec_list):.5f}')
        
        if self.task == 'ternary':
            logger.info(f'Class C - F1: {np.mean(c_f1_list):.5f}, '
                       f'Precision: {np.mean(c_prec_list):.5f}, Recall: {np.mean(c_rec_list):.5f}')

    def _train(self, epoch: int):
        """Training loop for one epoch"""
        loss_list = []
        loss_pre_list = []
        self.model.train()
        
        pbar = tqdm(self.train_dataloader, 
                   bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        
        for batch in pbar:
            _ = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
            
            output = self.model(**inputs)
            pred = output['pred'] if isinstance(output, dict) else output
            
            loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        metrics = self.evaluator.compute()
        
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list):.5f}")
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, '
                   f'Macro F1: {metrics["macro_f1"]:.5f}, '
                   f'Macro Prec: {metrics["macro_prec"]:.5f}, '
                   f'Macro Rec: {metrics["macro_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, '
                   f'A Prec: {metrics["a_prec"]:.5f}, A Rec: {metrics["a_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, '
                   f'B Prec: {metrics["b_prec"]:.5f}, B Rec: {metrics["b_rec"]:.5f}')
        
        if self.task == 'ternary':
            logger.info(f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, '
                       f'C Prec: {metrics["c_prec"]:.5f}, C Rec: {metrics["c_rec"]:.5f}')
    
    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
        """Validation loop"""
        loss_list = []
        self.model.eval()
        
        if split == 'valid' and final:
            raise ValueError('final flag only supports test split')
        
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                pred = output['pred'] if isinstance(output, dict) else output
                loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        
        metrics = self.evaluator.compute()
        
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, "
                   f"Macro F1: {metrics['macro_f1']:.5f}, "
                   f"Macro Prec: {metrics['macro_prec']:.5f}, "
                   f"Macro Rec: {metrics['macro_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, "
                   f"A Prec: {metrics['a_prec']:.5f}, A Rec: {metrics['a_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, "
                   f"B Prec: {metrics['b_prec']:.5f}, B Rec: {metrics['b_rec']:.5f}")
        
        if self.task == 'ternary':
            logger.info(f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, "
                       f"C Prec: {metrics['c_prec']:.5f}, C Rec: {metrics['c_rec']:.5f}")
        
        if use_earlystop:
            if self.task == 'binary':
                self.earlystopping(metrics['acc'], self.model)
            else:
                raise ValueError('Early stopping only implemented for binary task')
        
        return metrics


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="HateMM_MoRE")
def main(cfg: DictConfig):
    """Main entry point"""
    log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()