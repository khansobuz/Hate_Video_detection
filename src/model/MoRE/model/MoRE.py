import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange

from .submodule import LearnablePositionalEncoding, check_shape, AttentivePooling

class ModalityFFN(nn.Module):
    def __init__(self, hid_dim):
        super(ModalityFFN, self).__init__()
        self.modailty_ffn = nn.Sequential(
            nn.LazyLinear(hid_dim), 
            nn.ReLU(), 
            nn.LazyLinear(hid_dim)
        )

    def forward(self, x, pos_x, neg_x):
        x = self.modailty_ffn(x)
        pos_x = self.modailty_ffn(pos_x)
        neg_x = self.modailty_ffn(neg_x)
        return x, pos_x, neg_x


class ModalityExpert(nn.Module):
    def __init__(self, hid_dim, dropout, num_head, alpha, ablation):
        super(ModalityExpert, self).__init__()
        self.ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
        self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.alpha = alpha
        self.ablation = ablation

    def forward(self, query, pos, neg):
        # Ensure query is 3D: [B, Lq, D]
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query = self.ffn(query)

        # Ensure pos/neg are 3D: [B, N*L, D]
        if pos.dim() == 4:
            B, N, L, D = pos.shape
            pos = pos.view(B, N*L, D)
        pos = self.ffn(pos)

        if neg.dim() == 4:
            B, N, L, D = neg.shape
            neg = neg.view(B, N*L, D)
        neg = self.ffn(neg)

        # MultiheadAttention: query [B, Lq, D], key/value [B, Lk, D]
        pos_attn, _ = self.pos_attn(query, pos, pos)
        neg_attn, _ = self.neg_attn(query, neg, neg)

        ret = self.alpha * pos_attn + (1 - self.alpha) * neg_attn + query
        return ret.squeeze(1)  # remove sequence dimension


class MoRE(nn.Module):
    def __init__(self, text_encoder, fea_dim=768, dropout=0.2, num_head=8, alpha=0.5, delta=0.25, num_epoch=20, ablation='No', loss='No', **kargs):
        super(MoRE, self).__init__()

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

        self.router = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(3), nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(nn.LazyLinear(200), nn.ReLU(), nn.Dropout(dropout), nn.Linear(200, 2))

        self.text_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
        self.vision_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
        self.audio_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))

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

        # Positional encoding for vision features
        vision_fea = self.positional_encoding(vision_fea)
        frame_sim_pos_fea = self.positional_encoding(frame_sim_pos_fea)
        frame_sim_neg_fea = self.positional_encoding(frame_sim_neg_fea)

        # Modality experts
        text_fea_aug = self.text_expert(text_fea, text_sim_pos_fea, text_sim_neg_fea)
        vision_fea_aug = self.vision_expert(vision_fea, frame_sim_pos_fea, frame_sim_neg_fea)
        audio_fea_aug = self.audio_expert(audio_fea, mfcc_sim_pos_fea, mfcc_sim_neg_fea)

        # Collapse sequence dimension for router
        text_fea_ = text_fea if text_fea.dim() == 2 else text_fea.mean(dim=1)
        vision_fea_ = vision_fea.mean(dim=1)
        audio_fea_ = audio_fea if audio_fea.dim() == 2 else audio_fea.mean(dim=1)

        router_fea = torch.cat([text_fea_, vision_fea_, audio_fea_], dim=-1)
        weight = self.router(router_fea).squeeze(1)

        # Pool features for final prediction
        text_fea_aug = text_fea_aug.mean(dim=1)
        vision_fea_aug = self.vision_pooler(vision_fea_aug)
        audio_fea_aug = audio_fea_aug.mean(dim=1)

        text_pred = self.text_preditor(text_fea_aug)
        vision_pred = self.vision_preditor(vision_fea_aug)
        audio_pred = self.audio_preditor(audio_fea_aug)

        # Combine features using router weights
        if self.ablation == 'w/o-router':
            fea = (text_fea_aug + vision_fea_aug + audio_fea_aug) / 3
        else:
            fea = (
                text_fea_aug * weight[:, 0].unsqueeze(1) +
                vision_fea_aug * weight[:, 1].unsqueeze(1) +
                audio_fea_aug * weight[:, 2].unsqueeze(1)
            )

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
        l_join = min(1-f_epo, delta) * l_exp + max(f_epo, 1-delta) * l_mix

        return l_join, l_mix
