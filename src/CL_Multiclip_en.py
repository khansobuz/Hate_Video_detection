"""
Complete Anti-Overfitting Novel Four-Step Method for MultiHateClip EN Dataset
FAME + R³GC + CMEL + UMAF with Strong Regularization
"""

import sys
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
from functools import partial
from collections import defaultdict

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
import hydra
from omegaconf import DictConfig, OmegaConf

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mixup_data(x, y, alpha=0.3):
    """Mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ============================================================================
# DATASET CLASSES
# ============================================================================

class MHClipEN_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        
    def _get_data(self, fold: int, split: str, task: str):
        data_path = Path('data/MultiHateClip/en')
        data_file = data_path / f'{split}.csv'
        label_file = data_path / 'label.jsonl'
        
        if not data_file.exists():
            raise FileNotFoundError(f'Data file not found: {data_file}')
        if not label_file.exists():
            raise FileNotFoundError(f'Label file not found: {label_file}')
        
        df = pd.read_csv(data_file, header=None, names=['Video_ID'])
        labels_df = pd.read_json(label_file, lines=True)
        df = df.merge(labels_df, left_on='Video_ID', right_on='vid', how='left')
        
        if 'vid' in df.columns:
            df = df.drop(columns=['vid'])
        
        print(f"Loaded {split}.csv: {len(df)} rows, Label distribution: {df['label'].value_counts().to_dict()}")
        
        missing_labels = df['label'].isna().sum()
        if missing_labels > 0:
            print(f"WARNING: {missing_labels} videos missing labels")
            df = df.dropna(subset=['label'])
            df = df.reset_index(drop=True)
        
        return df

class MHClipEN_Novel_Dataset(MHClipEN_Dataset):
    def __init__(self, fold: int, split: str, task: str, num_pos=30, num_neg=90, **kwargs):
        super().__init__()
        fea_path = Path('data/MultiHateClip/en/fea')
        sim_path = Path('data/MultiHateClip/en/retrieval')
        
        self.data = self._get_data(fold, split, task)
        self.mfcc_fea = torch.load(fea_path / 'fea_audio_mfcc.pt', weights_only=True)
        self.text_fea = torch.load(fea_path / 'fea_title_trans_bert-base-uncased.pt', weights_only=True)
        self.frame_fea = torch.load(fea_path / 'fea_frames_16_google-vit-base-16-224.pt', weights_only=True)
        self.sim_all_sim = pd.read_json(sim_path / 'all_modal.jsonl', lines=True)
        
        self.num_pos = num_pos
        self.num_neg = num_neg
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = item['label'] if 'label' in item else item.iloc[1]
        
        if pd.isna(label):
            raise ValueError(f"Label is NaN for index {idx}")
        
        label = int(float(label))
        vid = item['Video_ID'] if 'Video_ID' in item else item.iloc[0]
        
        text_fea = self.text_fea[vid]
        vision_fea = self.frame_fea[vid]
        audio_fea = self.mfcc_fea[vid]
        
        sim_data = self.sim_all_sim[self.sim_all_sim['vid'] == vid].iloc[0]['similarities']
        all_sim_pos_vids = sim_data[0]['vid'][:self.num_pos]
        all_sim_neg_vids = sim_data[1]['vid'][:self.num_neg]
        
        text_sim_pos_fea = torch.stack([self.text_fea[key] for key in all_sim_pos_vids])
        text_sim_neg_fea = torch.stack([self.text_fea[key] for key in all_sim_neg_vids])
        vision_sim_pos_fea = torch.stack([self.frame_fea[key] for key in all_sim_pos_vids])
        vision_sim_neg_fea = torch.stack([self.frame_fea[key] for key in all_sim_neg_vids])
        audio_sim_pos_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_pos_vids])
        audio_sim_neg_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_neg_vids])
        
        return {
            'vid': vid, 'label': torch.tensor(label, dtype=torch.long), 'text_fea': text_fea, 
            'vision_fea': vision_fea, 'audio_fea': audio_fea, 'text_sim_pos_fea': text_sim_pos_fea,
            'text_sim_neg_fea': text_sim_neg_fea, 'vision_sim_pos_fea': vision_sim_pos_fea,
            'vision_sim_neg_fea': vision_sim_neg_fea, 'audio_sim_pos_fea': audio_sim_pos_fea,
            'audio_sim_neg_fea': audio_sim_neg_fea, 'neighbor_vids': all_sim_pos_vids + all_sim_neg_vids
        }

class MHClipEN_Novel_Collator:
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
            'neighbor_vids': [item['neighbor_vids'] for item in batch]
        }

 
"""
Complete Anti-Overfitting Novel Four-Step Method for 85%+ Accuracy
FAME + R³GC + CMEL + UMAF with Strong Regularization
"""

 

# ============================================================================
# ANTI-OVERFITTING COMPONENTS
# ============================================================================

class DropPath(nn.Module):
    """Stochastic Depth - drops entire residual branches"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class BidirectionalCrossModalAttention(nn.Module):
    """Enhanced cross-modal attention with strong regularization"""
    def __init__(self, dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.text_to_vision = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.vision_to_text = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_audio = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_to_text = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.vision_to_audio = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_to_vision = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.text_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.vision_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        
        self.norm_text = nn.LayerNorm(dim)
        self.norm_vision = nn.LayerNorm(dim)
        self.norm_audio = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_prob=0.1)
        
    def forward(self, text, vision, audio):
        t2v, _ = self.text_to_vision(text.unsqueeze(1), vision.unsqueeze(1), vision.unsqueeze(1))
        v2t, _ = self.vision_to_text(vision.unsqueeze(1), text.unsqueeze(1), text.unsqueeze(1))
        t2v, v2t = t2v.squeeze(1), v2t.squeeze(1)
        
        gate_t = self.text_gate(torch.cat([text, v2t], dim=-1))
        text_enhanced = self.norm_text(text + self.drop_path(gate_t * (t2v + v2t)))
        
        t2a, _ = self.text_to_audio(text_enhanced.unsqueeze(1), audio.unsqueeze(1), audio.unsqueeze(1))
        a2t, _ = self.audio_to_text(audio.unsqueeze(1), text_enhanced.unsqueeze(1), text_enhanced.unsqueeze(1))
        t2a, a2t = t2a.squeeze(1), a2t.squeeze(1)
        
        gate_t2 = self.text_gate(torch.cat([text_enhanced, a2t], dim=-1))
        text_final = self.norm_text(text_enhanced + self.drop_path(gate_t2 * (t2a + a2t)))
        
        v2a, _ = self.vision_to_audio(vision.unsqueeze(1), audio.unsqueeze(1), audio.unsqueeze(1))
        a2v, _ = self.audio_to_vision(audio.unsqueeze(1), vision.unsqueeze(1), vision.unsqueeze(1))
        v2a, a2v = v2a.squeeze(1), a2v.squeeze(1)
        
        gate_v = self.vision_gate(torch.cat([vision, a2v + v2t], dim=-1))
        vision_final = self.norm_vision(vision + self.drop_path(gate_v * (v2a + a2v + v2t)))
        
        gate_a = self.audio_gate(torch.cat([audio, a2t + a2v], dim=-1))
        audio_final = self.norm_audio(audio + self.drop_path(gate_a * (a2t + a2v)))
        
        return text_final, vision_final, audio_final

class EnhancedGraphAttention(nn.Module):
    """Graph attention with dropout"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(out_dim)
        
    def forward(self, node_feat, neighbor_feats):
        B, D = node_feat.shape
        N, _ = neighbor_feats.shape
        Q = self.query(node_feat).view(B, self.num_heads, self.head_dim)
        K = self.key(neighbor_feats).view(N, self.num_heads, self.head_dim)
        V = self.value(neighbor_feats).view(N, self.num_heads, self.head_dim)
        scores = torch.einsum('bhd,nhd->bhn', Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhn,nhd->bhd', attn, V)
        out = out.reshape(B, -1)
        out = self.out_proj(out)
        if D != out.shape[-1]:
            node_feat_proj = nn.Linear(D, out.shape[-1]).to(out.device)(node_feat)
        else:
            node_feat_proj = node_feat
        out = self.norm(out + node_feat_proj)
        out = self.norm2(out + self.ffn(out))
        return out

# ============================================================================
# STEP 1: FAME
# ============================================================================

class FAME(nn.Module):
    """Frozen Adaptive Multimodal Encoding"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    def forward(self, x):
        adapted = self.adapter(x)
        if x.shape[-1] == self.hidden_dim:
            return adapted + x
        else:
            return adapted

# ============================================================================
# STEP 2: R³GC with Edge Dropout (Anti-Overfitting)
# ============================================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.enhanced_gat = EnhancedGraphAttention(in_dim, out_dim, num_heads, dropout)
    def forward(self, node_feat, neighbor_feats):
        return self.enhanced_gat(node_feat, neighbor_feats)

class R3GC(nn.Module):
    """R³GC with edge dropout to prevent memorizing neighbors"""
    def __init__(self, feat_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.3, edge_dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.edge_dropout = edge_dropout
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(feat_dim if i == 0 else hidden_dim, hidden_dim, num_heads, dropout)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop_path = DropPath(drop_prob=0.1)
        
    def forward(self, node_feat, pos_neighbor_feats, neg_neighbor_feats):
        # Edge dropout: randomly drop neighbors during training
        if self.training and self.edge_dropout > 0:
            keep_ratio = 1.0 - self.edge_dropout
            n_pos_keep = max(1, int(pos_neighbor_feats.shape[0] * keep_ratio))
            pos_indices = torch.randperm(pos_neighbor_feats.shape[0])[:n_pos_keep]
            pos_neighbor_feats = pos_neighbor_feats[pos_indices]
            
            n_neg_keep = max(1, int(neg_neighbor_feats.shape[0] * keep_ratio))
            neg_indices = torch.randperm(neg_neighbor_feats.shape[0])[:n_neg_keep]
            neg_neighbor_feats = neg_neighbor_feats[neg_indices]
        
        all_neighbors = torch.cat([pos_neighbor_feats, neg_neighbor_feats], dim=0)
        x = node_feat
        
        for gat, norm in zip(self.gat_layers, self.norms):
            x_new = gat(x, all_neighbors)
            x_new = self.drop_path(x_new)
            x = norm(x_new + x) if x.shape == x_new.shape else norm(x_new)
        
        return x

# ============================================================================
# STEP 3: CMEL with Hard Sample Mining
# ============================================================================

class MemoryBank:
    """Memory bank with hard sample mining"""
    def __init__(self, memory_size=1000, num_classes=2):
        self.memory_size = memory_size
        self.num_classes = num_classes
        self.memory = {i: {'feats': [], 'labels': [], 'vids': [], 'losses': []} for i in range(num_classes)}
        self.per_class_size = memory_size // num_classes
        
    def add_samples(self, features, labels, vids, losses=None):
        if losses is None:
            losses = [1.0] * len(features)
            
        for feat, label, vid, loss in zip(features, labels, vids, losses):
            label_int = label.item() if torch.is_tensor(label) else int(label)
            loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            
            if len(self.memory[label_int]['feats']) < self.per_class_size:
                self.memory[label_int]['feats'].append(feat.detach().cpu())
                self.memory[label_int]['labels'].append(label_int)
                self.memory[label_int]['vids'].append(vid)
                self.memory[label_int]['losses'].append(loss_val)
            else:
                idx = np.argmin(self.memory[label_int]['losses'])
                if loss_val > self.memory[label_int]['losses'][idx]:
                    self.memory[label_int]['feats'][idx] = feat.detach().cpu()
                    self.memory[label_int]['labels'][idx] = label_int
                    self.memory[label_int]['vids'][idx] = vid
                    self.memory[label_int]['losses'][idx] = loss_val
                    
    def get_memory_batch(self, batch_size, device):
        if self.is_empty():
            return None, None
        samples_per_class = batch_size // self.num_classes
        all_feats, all_labels = [], []
        
        for cls in range(self.num_classes):
            if len(self.memory[cls]['feats']) > 0:
                n_samples = min(samples_per_class, len(self.memory[cls]['feats']))
                losses = np.array(self.memory[cls]['losses'])
                if losses.sum() > 0:
                    probs = losses / losses.sum()
                    indices = np.random.choice(len(self.memory[cls]['feats']), n_samples, replace=False, p=probs)
                else:
                    indices = np.random.choice(len(self.memory[cls]['feats']), n_samples, replace=False)
                
                all_feats.extend([self.memory[cls]['feats'][i] for i in indices])
                all_labels.extend([self.memory[cls]['labels'][i] for i in indices])
                
        if len(all_feats) == 0:
            return None, None
        return torch.stack(all_feats).to(device), torch.tensor(all_labels, dtype=torch.long).to(device)
    
    def is_empty(self):
        return all(len(self.memory[i]['feats']) == 0 for i in range(self.num_classes))

class CMEL(nn.Module):
    """Continual Memory-Enhanced Learning"""
    def __init__(self, feat_dim, memory_size=1000, num_classes=2):
        super().__init__()
        self.memory_bank = MemoryBank(memory_size, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim)
        )
    def forward(self, features):
        return self.projection(features)
    def update_memory(self, features, labels, vids, losses=None):
        self.memory_bank.add_samples(features, labels, vids, losses)
    def get_memory_batch(self, batch_size, device):
        return self.memory_bank.get_memory_batch(batch_size, device)

# ============================================================================
# STEP 4: UMAF
# ============================================================================

class UMAF(nn.Module):
    """Uncertainty-Aware Modality Attentive Fusion"""
    def __init__(self, feat_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.text_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes)
        )
        self.vision_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes)
        )
        self.audio_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes)
        )
        self.uncertainty_attn = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )
    def compute_uncertainty(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    def forward(self, text_feat, vision_feat, audio_feat):
        text_logits = self.text_predictor(text_feat)
        vision_logits = self.vision_predictor(vision_feat)
        audio_logits = self.audio_predictor(audio_feat)
        text_unc = self.compute_uncertainty(text_logits)
        vision_unc = self.compute_uncertainty(vision_logits)
        audio_unc = self.compute_uncertainty(audio_logits)
        uncertainties = torch.stack([text_unc, vision_unc, audio_unc], dim=-1)
        inv_uncertainties = 1.0 / (uncertainties + 1e-6)
        attn_weights = self.uncertainty_attn(inv_uncertainties)
        fused_feat = torch.cat([
            text_feat * attn_weights[:, 0:1],
            vision_feat * attn_weights[:, 1:2],
            audio_feat * attn_weights[:, 2:3]
        ], dim=-1)
        final_logits = self.final_classifier(fused_feat)
        return {
            'logits': final_logits,
            'text_logits': text_logits,
            'vision_logits': vision_logits,
            'audio_logits': audio_logits,
            'attn_weights': attn_weights
        }

# ============================================================================
# MAIN MODEL
# ============================================================================

class NovelCLModel(nn.Module):
    """Anti-Overfitting Four-Step Method"""
    name = 'NovelCL_AntiOverfit'
    def __init__(self, text_encoder, fea_dim=128, dropout=0.3, num_head=4, 
                 memory_size=1000, num_classes=2, num_epoch=100, label_smoothing=0.15, 
                 use_mixup=True, mixup_alpha=0.3, **kwargs):
        super().__init__()
        self.text_proj = nn.Linear(768, fea_dim)
        self.vision_proj = nn.Linear(768, fea_dim)
        self.audio_proj = nn.Linear(128, fea_dim)
        
        self.text_fame = FAME(fea_dim, fea_dim, dropout)
        self.vision_fame = FAME(fea_dim, fea_dim, dropout)
        self.audio_fame = FAME(fea_dim, fea_dim, dropout)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 16, 768))
        
        self.cross_modal_attn = BidirectionalCrossModalAttention(fea_dim, num_heads=num_head, dropout=dropout)
        
        # 2 layers (middle ground between 1 and 3)
        self.text_graph = R3GC(fea_dim, fea_dim, num_layers=2, num_heads=num_head, dropout=dropout, edge_dropout=0.25)
        self.vision_graph = R3GC(fea_dim, fea_dim, num_layers=2, num_heads=num_head, dropout=dropout, edge_dropout=0.25)
        self.audio_graph = R3GC(fea_dim, fea_dim, num_layers=2, num_heads=num_head, dropout=dropout, edge_dropout=0.25)
        
        self.cmel = CMEL(fea_dim, memory_size, num_classes)
        self.umaf = UMAF(fea_dim, num_classes, dropout)
        
        self.num_classes = num_classes
        self.total_epoch = num_epoch
        self.label_smoothing = label_smoothing
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
    def forward(self, **inputs):
        text_fea = inputs['text_fea']
        audio_fea = inputs['audio_fea']
        vision_fea = inputs['vision_fea']
        text_sim_pos_fea = inputs['text_sim_pos_fea']
        text_sim_neg_fea = inputs['text_sim_neg_fea']
        vision_sim_pos_fea = inputs['vision_sim_pos_fea']
        vision_sim_neg_fea = inputs['vision_sim_neg_fea']
        audio_sim_pos_fea = inputs['audio_sim_pos_fea']
        audio_sim_neg_fea = inputs['audio_sim_neg_fea']
        
        B = text_fea.shape[0]
        
        if text_fea.dim() == 2:
            text_fea = text_fea.unsqueeze(1)
        if audio_fea.dim() == 2:
            audio_fea = audio_fea.unsqueeze(1)
            
        vision_fea = vision_fea + self.pos_encoding[:, :vision_fea.size(1), :]
        
        N_pos = vision_sim_pos_fea.shape[1]
        N_neg = vision_sim_neg_fea.shape[1]
        L = vision_sim_pos_fea.shape[2]
        D = vision_sim_pos_fea.shape[3]
        
        vision_sim_pos_fea = vision_sim_pos_fea.reshape(B*N_pos, L, D)
        vision_sim_pos_fea = vision_sim_pos_fea + self.pos_encoding[:, :L, :]
        vision_sim_pos_fea = vision_sim_pos_fea.reshape(B, N_pos, L, D).mean(dim=2)
        
        vision_sim_neg_fea = vision_sim_neg_fea.reshape(B*N_neg, L, D)
        vision_sim_neg_fea = vision_sim_neg_fea + self.pos_encoding[:, :L, :]
        vision_sim_neg_fea = vision_sim_neg_fea.reshape(B, N_neg, L, D).mean(dim=2)
        
        text_fea_pooled = text_fea.mean(dim=1)
        vision_fea_pooled = vision_fea.mean(dim=1)
        audio_fea_pooled = audio_fea.mean(dim=1)
        
        text_sim_pos_pooled = text_sim_pos_fea.mean(dim=1)
        text_sim_neg_pooled = text_sim_neg_fea.mean(dim=1)
        audio_sim_pos_pooled = audio_sim_pos_fea.mean(dim=1)
        audio_sim_neg_pooled = audio_sim_neg_fea.mean(dim=1)
        
        text_fea_pooled = self.text_proj(text_fea_pooled)
        vision_fea_pooled = self.vision_proj(vision_fea_pooled)
        audio_fea_pooled = self.audio_proj(audio_fea_pooled)
        
        text_sim_pos_pooled = self.text_proj(text_sim_pos_pooled)
        text_sim_neg_pooled = self.text_proj(text_sim_neg_pooled)
        vision_sim_pos_pooled = self.vision_proj(vision_sim_pos_fea.mean(dim=1))
        vision_sim_neg_pooled = self.vision_proj(vision_sim_neg_fea.mean(dim=1))
        audio_sim_pos_pooled = self.audio_proj(audio_sim_pos_pooled)
        audio_sim_neg_pooled = self.audio_proj(audio_sim_neg_pooled)
        
        text_adapted = self.text_fame(text_fea_pooled)
        vision_adapted = self.vision_fame(vision_fea_pooled)
        audio_adapted = self.audio_fame(audio_fea_pooled)
        
        text_sim_pos_adapted = self.text_fame(text_sim_pos_pooled)
        text_sim_neg_adapted = self.text_fame(text_sim_neg_pooled)
        vision_sim_pos_adapted = self.vision_fame(vision_sim_pos_pooled)
        vision_sim_neg_adapted = self.vision_fame(vision_sim_neg_pooled)
        audio_sim_pos_adapted = self.audio_fame(audio_sim_pos_pooled)
        audio_sim_neg_adapted = self.audio_fame(audio_sim_neg_pooled)
        
        text_adapted, vision_adapted, audio_adapted = self.cross_modal_attn(
            text_adapted, vision_adapted, audio_adapted
        )
        
        text_graph_feat = self.text_graph(text_adapted, text_sim_pos_adapted, text_sim_neg_adapted)
        vision_graph_feat = self.vision_graph(vision_adapted, vision_sim_pos_adapted, vision_sim_neg_adapted)
        audio_graph_feat = self.audio_graph(audio_adapted, audio_sim_pos_adapted, audio_sim_neg_adapted)
        
        text_cmel = self.cmel(text_graph_feat)
        vision_cmel = self.cmel(vision_graph_feat)
        audio_cmel = self.cmel(audio_graph_feat)
        
        outputs = self.umaf(text_cmel, vision_cmel, audio_cmel)
        
        outputs['text_feat'] = text_cmel
        outputs['vision_feat'] = vision_cmel
        outputs['audio_feat'] = audio_cmel
        outputs['combined_feat'] = (text_cmel + vision_cmel + audio_cmel) / 3.0
        
        return outputs
    
    def calculate_loss(self, **inputs):
        logits = inputs['logits']
        text_logits = inputs['text_logits']
        vision_logits = inputs['vision_logits']
        audio_logits = inputs['audio_logits']
        labels = inputs['label']
        epoch = inputs.get('epoch', 0)
        
        if 'label_a' in inputs and 'label_b' in inputs and 'lam' in inputs:
            label_a, label_b, lam = inputs['label_a'], inputs['label_b'], inputs['lam']
            main_loss = lam * F.cross_entropy(logits, label_a, label_smoothing=self.label_smoothing) + \
                       (1 - lam) * F.cross_entropy(logits, label_b, label_smoothing=self.label_smoothing)
            text_loss = lam * F.cross_entropy(text_logits, label_a) + (1 - lam) * F.cross_entropy(text_logits, label_b)
            vision_loss = lam * F.cross_entropy(vision_logits, label_a) + (1 - lam) * F.cross_entropy(vision_logits, label_b)
            audio_loss = lam * F.cross_entropy(audio_logits, label_a) + (1 - lam) * F.cross_entropy(audio_logits, label_b)
        else:
            main_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            text_loss = F.cross_entropy(text_logits, labels)
            vision_loss = F.cross_entropy(vision_logits, labels)
            audio_loss = F.cross_entropy(audio_logits, labels)
        
        modality_loss = (text_loss + vision_loss + audio_loss) / 3.0
        alpha = min(1.0, epoch / (self.total_epoch * 0.5))
        total_loss = alpha * main_loss + (1 - alpha) * modality_loss
        
        return total_loss, main_loss
    
    def update_memory(self, features, labels, vids, losses=None):
        self.cmel.update_memory(features, labels, vids, losses)
    
    def get_memory_batch(self, batch_size, device):
        return self.cmel.get_memory_batch(batch_size, device)

# ============================================================================
# METRICS
# ============================================================================

class BinaryClassificationMetric:
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

class EarlyStopping:
    """Early stopping based on validation accuracy"""
    def __init__(self, patience=15, path='best_model.pth', delta=0.001):
        self.patience = patience
        self.path = path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.inf
    
    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"{Fore.YELLOW}EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
    
    def save_checkpoint(self, val_acc, model):
        torch.save(model.state_dict(), self.path)
        self.best_metric = val_acc
        logger.info(f"{Fore.GREEN}✓ Saved best model | Val Acc: {val_acc:.5f}")

def get_optimizer(model, name='AdamW', lr=0.0002, weight_decay=0.0001, **kwargs):
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {name} not supported')

def get_scheduler(optimizer, name='CosineAnnealingLR', num_epochs=100, **kwargs):
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif name == 'DummyLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f'Scheduler {name} not supported')

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = cfg.task
        if cfg.task == 'binary':
            self.evaluator = BinaryClassificationMetric(self.device)
        else:
            raise ValueError('Only binary task supported')
        self.type = cfg.type
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        if cfg.type == 'default':
            self.dataset_range = ['default']
        else:
            raise ValueError('experiment type not supported')
        self.collator = MHClipEN_Novel_Collator(**cfg.data)
        self.use_memory = cfg.get('use_memory', True)
        self.memory_sample_ratio = cfg.get('memory_sample_ratio', 0.12)
        self.use_mixup = cfg.para.get('use_mixup', True)
        self.mixup_prob = 0.55  # 55% of batches (middle ground)
        
    def _reset(self, cfg, fold, type):
        train_dataset = MHClipEN_Novel_Dataset(task=cfg.task, fold=fold, split='train', **cfg.data)
        test_dataset = MHClipEN_Novel_Dataset(task=cfg.task, fold=fold, split='test', **cfg.data)
        if cfg.task == 'binary':
            valid_dataset = MHClipEN_Novel_Dataset(task=cfg.task, fold=fold, split='valid', **cfg.data)
        
        # Fix Windows multiprocessing error: set num_workers=0
        num_workers = 0
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
            num_workers=num_workers, shuffle=True, generator=self.generator)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
            num_workers=num_workers, shuffle=False, generator=self.generator)
        if cfg.task == 'binary':
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
                num_workers=num_workers, shuffle=False, generator=self.generator)
        
        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = NovelCLModel(**dict(cfg.para))
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, num_epochs=cfg.num_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path / 'best_model.pth')
        
    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        a_f1_list, a_prec_list, a_rec_list = [], [], []
        b_f1_list, b_prec_list, b_rec_list = [], [], []
        
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'{Fore.CYAN}Current fold: {fold} - Anti-Overfitting Novel CL Method')
            
            for epoch in range(self.num_epoch):
                logger.info(f'{Fore.CYAN}Epoch {epoch}/{self.num_epoch}')
                self._train(epoch=epoch)
                
                if self.task == 'binary':
                    val_metrics = self._valid(split='valid', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                        break
                    self._valid(split='test', epoch=epoch)
                
                # Step scheduler per epoch
                self.scheduler.step()
            
            logger.info(f'{Fore.RED}Best results for fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path / 'best_model.pth', weights_only=False))
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
        
        logger.info(f'{Fore.MAGENTA}=== FINAL RESULTS (Anti-Overfitting) ===')
        logger.info(f'{Fore.MAGENTA}Avg Acc: {np.mean(acc_list):.5f}±{np.std(acc_list):.5f}, F1: {np.mean(f1_list):.5f}±{np.std(f1_list):.5f}')
        logger.info(f'{Fore.MAGENTA}Avg Precision: {np.mean(prec_list):.5f}±{np.std(prec_list):.5f}, Recall: {np.mean(rec_list):.5f}±{np.std(rec_list):.5f}')
        logger.info(f'{Fore.MAGENTA}Class 0 (Non-Hate) - F1: {np.mean(a_f1_list):.5f}±{np.std(a_f1_list):.5f}, Prec: {np.mean(a_prec_list):.5f}±{np.std(a_prec_list):.5f}, Rec: {np.mean(a_rec_list):.5f}±{np.std(a_rec_list):.5f}')
        logger.info(f'{Fore.MAGENTA}Class 1 (Hate) - F1: {np.mean(b_f1_list):.5f}±{np.std(b_f1_list):.5f}, Prec: {np.mean(b_prec_list):.5f}±{np.std(b_prec_list):.5f}, Rec: {np.mean(b_rec_list):.5f}±{np.std(b_rec_list):.5f}')
        
    def _train(self, epoch: int):
        loss_list = []
        loss_main_list = []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        
        for batch in pbar:
            vids = batch.pop('vids')
            neighbor_vids = batch.pop('neighbor_vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
            
            use_mixup_batch = self.use_mixup and np.random.rand() < self.mixup_prob
            
            if use_mixup_batch:
                outputs_pre = self.model(**inputs)
                combined_feat = outputs_pre['combined_feat']
                mixed_feat, labels_a, labels_b, lam = mixup_data(combined_feat, labels, self.model.mixup_alpha)
                outputs = self.model.umaf(mixed_feat, mixed_feat, mixed_feat)
                outputs['combined_feat'] = mixed_feat
                outputs.update({
                    'text_feat': outputs_pre['text_feat'],
                    'vision_feat': outputs_pre['vision_feat'],
                    'audio_feat': outputs_pre['audio_feat']
                })
                loss, loss_main = self.model.calculate_loss(
                    **outputs, label=labels, label_a=labels_a, label_b=labels_b, lam=lam, epoch=epoch
                )
            else:
                outputs = self.model(**inputs)
                loss, loss_main = self.model.calculate_loss(**outputs, label=labels, epoch=epoch)
            
            pred = outputs['logits']
            
            if self.use_memory and not self.model.cmel.memory_bank.is_empty():
                memory_batch_size = int(self.batch_size * self.memory_sample_ratio)
                mem_feats, mem_labels = self.model.get_memory_batch(memory_batch_size, self.device)
                
                if mem_feats is not None:
                    mem_outputs = self.model.umaf(mem_feats, mem_feats, mem_feats)
                    mem_loss = F.cross_entropy(mem_outputs['logits'], mem_labels)
                    loss = loss + 0.3 * mem_loss
            
            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_main_list.append(loss_main.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.use_memory:
                with torch.no_grad():
                    combined_feats = (outputs['text_feat'] + outputs['vision_feat'] + outputs['audio_feat']) / 3.0
                    sample_losses = F.cross_entropy(pred, labels, reduction='none')
                    self.model.update_memory(combined_feats, labels, vids, sample_losses)
        
        metrics = self.evaluator.compute()
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"{Fore.BLUE}Train Loss: {np.mean(loss_list):.5f} (Main: {np.mean(loss_main_list):.5f}) | LR: {current_lr:.6f}")
        logger.info(f'{Fore.BLUE}Train Acc: {metrics["acc"]:.5f}, F1: {metrics["macro_f1"]:.5f}, Prec: {metrics["macro_prec"]:.5f}, Rec: {metrics["macro_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Class 0: F1={metrics["a_f1"]:.5f}, Prec={metrics["a_prec"]:.5f}, Rec={metrics["a_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Class 1: F1={metrics["b_f1"]:.5f}, Prec={metrics["b_prec"]:.5f}, Rec={metrics["b_rec"]:.5f}')
        
    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
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
            neighbor_vids = batch.pop('neighbor_vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = outputs['logits']
                loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        
        metrics = self.evaluator.compute()
        logger.info(f"{fcolor}{split_name} Loss: {np.mean(loss_list):.5f}")
        logger.info(f"{fcolor}{split_name} Acc: {metrics['acc']:.5f}, F1: {metrics['macro_f1']:.5f}, Prec: {metrics['macro_prec']:.5f}, Rec: {metrics['macro_rec']:.5f}")
        logger.info(f"{fcolor}Class 0: F1={metrics['a_f1']:.5f}, Prec={metrics['a_prec']:.5f}, Rec={metrics['a_rec']:.5f}")
        logger.info(f"{fcolor}Class 1: F1={metrics['b_f1']:.5f}, Prec={metrics['b_prec']:.5f}, Rec={metrics['b_rec']:.5f}")
        
        if use_earlystop:
            if self.task == 'binary':
                self.earlystopping(metrics['acc'], self.model)
        
        return metrics

# ============================================================================
# MAIN
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')
    log_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    
    logger.info(f"{Fore.CYAN}{'='*80}")
    logger.info(f"{Fore.CYAN}Anti-Overfitting Novel Four-Step Method for 85%+ Accuracy")
    logger.info(f"{Fore.CYAN}{'='*80}")
    logger.info(OmegaConf.to_yaml(cfg))
    
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == '__main__':
    main()