"""
BASELINE 2: Learning to Prompt (L2P)
Complete implementation - just copy and run!

Usage:
    python L2P_baseline.py

This will train on HateMM → MHClip-EN → MHClip-ZH with L2P prompts
"""

import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from loguru import logger
import colorama
from colorama import Fore
from tqdm import tqdm
import pandas as pd


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# L2P Prompt Pool
# ============================================================================

class L2PPromptPool(nn.Module):
    """Learning to Prompt - Prompt Pool"""
    def __init__(self, pool_size=10, prompt_length=5, embed_dim=128, selection_size=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.selection_size = selection_size
        
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_length, embed_dim))
        self.keys = nn.Parameter(torch.randn(pool_size, embed_dim))
        nn.init.uniform_(self.prompts, -1.0, 1.0)
        nn.init.uniform_(self.keys, -1.0, 1.0)
    
    def forward(self, x):
        B, D = x.shape
        x_norm = F.normalize(x, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)
        similarity = torch.matmul(x_norm, keys_norm.t())
        _, top_idx = torch.topk(similarity, self.selection_size, dim=1)
        selected = self.prompts[top_idx]
        prompt_feat = selected.mean(dim=1).mean(dim=1)
        return x + prompt_feat
    
    def diversity_loss(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)
        similarity = torch.matmul(x_norm, keys_norm.t())
        return similarity.max(dim=1)[0].mean()


# ============================================================================
# L2P Continual Learning
# ============================================================================

class L2P_ContinualLearning:
    def __init__(self, seed=2024, eval_epochs=40):
        self.seed = seed
        self.eval_epochs = eval_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(seed)
        
        self.accuracy_matrix = []
        self.save_path = Path(f'L2P_results_{datetime.now().strftime("%m%d-%H%M%S")}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{Fore.CYAN}{'='*80}")
        logger.info(f"{Fore.CYAN}Learning to Prompt (L2P) Baseline")
        logger.info(f"{Fore.CYAN}Pool Size: 10, Prompt Length: 5")
        logger.info(f"{Fore.CYAN}{'='*80}\n")
    
    def train_task(self, task_id, task_name, script_name, config_name):
        """Train on a task with L2P"""
        logger.info(f"{Fore.YELLOW}Training Task {task_id + 1}: {task_name}")
        
        import importlib.util
        from omegaconf import OmegaConf
        
        spec = importlib.util.spec_from_file_location(f"task_{task_id}", script_name)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        base_dir = Path.cwd()
        for base_path in [base_dir / "src" / "config", base_dir / "config"]:
            config_path = base_path / f"{config_name}.yaml"
            if config_path.exists():
                break
        
        cfg = OmegaConf.load(config_path)
        cfg.seed = self.seed
        
        trainer = task_module.Trainer(cfg)
        
        # Load previous model
        if hasattr(self, 'previous_model_path'):
            original_reset = trainer._reset
            prev_path = self.previous_model_path
            def patched_reset(cfg, fold, type):
                result = original_reset(cfg, fold, type)
                state_dict = torch.load(prev_path, weights_only=False)
                trainer.model.load_state_dict(state_dict, strict=False)
                logger.info(f"{Fore.GREEN}✓ Loaded previous model")
                return result
            trainer._reset = patched_reset
        
        # Initialize model
        trainer._reset(cfg, 'default', cfg.type)
        
        # Create L2P pools
        feat_dim = 128
        self.text_l2p = L2PPromptPool(10, 5, feat_dim).to(self.device)
        self.vision_l2p = L2PPromptPool(10, 5, feat_dim).to(self.device)
        self.audio_l2p = L2PPromptPool(10, 5, feat_dim).to(self.device)
        
        # Load previous L2P if exists
        if hasattr(self, 'previous_l2p_path'):
            l2p_state = torch.load(self.previous_l2p_path, weights_only=False)
            self.text_l2p.load_state_dict(l2p_state['text'])
            self.vision_l2p.load_state_dict(l2p_state['vision'])
            self.audio_l2p.load_state_dict(l2p_state['audio'])
            logger.info(f"{Fore.GREEN}✓ Loaded previous L2P prompts")
        
        # Add L2P to optimizer
        l2p_params = list(self.text_l2p.parameters()) + list(self.vision_l2p.parameters()) + list(self.audio_l2p.parameters())
        trainer.optimizer.add_param_group({'params': l2p_params, 'lr': trainer.cfg.opt.lr})
        
        # Patch training
        original_train = trainer._train
        def train_with_l2p(epoch):
            loss_list = []
            trainer.model.train()
            for batch in tqdm(trainer.train_dataloader, desc=f'Train E{epoch}'):
                batch.pop('vids')
                batch.pop('neighbor_vids')
                inputs = {k: v.to(trainer.device) for k, v in batch.items()}
                labels = inputs.pop('labels')
                
                # Forward
                outputs = trainer.model(**inputs)
                
                # Apply L2P
                text_p = self.text_l2p(outputs['text_feat'])
                vision_p = self.vision_l2p(outputs['vision_feat'])
                audio_p = self.audio_l2p(outputs['audio_feat'])
                
                # Re-fuse
                l2p_out = trainer.model.umaf(text_p, vision_p, audio_p)
                
                # Loss
                loss = F.cross_entropy(l2p_out['logits'], labels, label_smoothing=trainer.model.label_smoothing)
                div_loss = (self.text_l2p.diversity_loss(outputs['text_feat']) + 
                           self.vision_l2p.diversity_loss(outputs['vision_feat']) + 
                           self.audio_l2p.diversity_loss(outputs['audio_feat'])) / 3.0
                loss = loss + 0.01 * div_loss
                
                # Backward
                _, preds = torch.max(l2p_out['logits'], 1)
                trainer.evaluator.update(preds, labels)
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.text_l2p.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.vision_l2p.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.audio_l2p.parameters(), 0.5)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            
            metrics = trainer.evaluator.compute()
            logger.info(f"{Fore.BLUE}E{epoch} | Loss: {np.mean(loss_list):.4f} | Acc: {metrics['acc']:.4f}")
        
        trainer._train = train_with_l2p
        
        # Train
        trainer.run()
        
        # Save
        model_path = self.save_path / f'model_task_{task_id}.pth'
        torch.save(trainer.model.state_dict(), model_path)
        self.previous_model_path = model_path
        
        l2p_path = self.save_path / f'l2p_task_{task_id}.pth'
        torch.save({'text': self.text_l2p.state_dict(), 'vision': self.vision_l2p.state_dict(), 'audio': self.audio_l2p.state_dict()}, l2p_path)
        self.previous_l2p_path = l2p_path
        
        logger.info(f"{Fore.GREEN}✓ Task {task_id + 1} completed\n")
    
    def evaluate_task(self, task_id, task_name, script_name, config_name):
        """Evaluate on a task"""
        logger.info(f"{Fore.YELLOW}Evaluating Task {task_id + 1}: {task_name}")
        
        import importlib.util
        from omegaconf import OmegaConf
        
        spec = importlib.util.spec_from_file_location(f"eval_{task_id}_{datetime.now().strftime('%H%M%S')}", script_name)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        base_dir = Path.cwd()
        for base_path in [base_dir / "src" / "config", base_dir / "config"]:
            config_path = base_path / f"{config_name}.yaml"
            if config_path.exists():
                break
        
        cfg = OmegaConf.load(config_path)
        cfg.seed = self.seed
        cfg.num_epoch = self.eval_epochs
        
        trainer = task_module.Trainer(cfg)
        
        # Load model
        original_reset = trainer._reset
        model_loaded = False
        def patched_reset(cfg, fold, type):
            nonlocal model_loaded
            result = original_reset(cfg, fold, type)
            if not model_loaded:
                state_dict = torch.load(self.previous_model_path, weights_only=False)
                trainer.model.load_state_dict(state_dict, strict=False)
                model_loaded = True
            return result
        trainer._reset = patched_reset
        trainer._reset(cfg, 'default', cfg.type)
        
        # Fine-tune with L2P
        for epoch in range(self.eval_epochs):
            trainer._train(epoch=epoch)
            if epoch % 10 == 0:
                # Eval with L2P
                trainer.model.eval()
                for batch in trainer.test_dataloader:
                    batch.pop('vids')
                    batch.pop('neighbor_vids')
                    inputs = {k: v.to(trainer.device) for k, v in batch.items()}
                    labels = inputs.pop('labels')
                    with torch.no_grad():
                        outputs = trainer.model(**inputs)
                        text_p = self.text_l2p(outputs['text_feat'])
                        vision_p = self.vision_l2p(outputs['vision_feat'])
                        audio_p = self.audio_l2p(outputs['audio_feat'])
                        l2p_out = trainer.model.umaf(text_p, vision_p, audio_p)
                        _, preds = torch.max(l2p_out['logits'], 1)
                        trainer.evaluator.update(preds, labels)
                metrics = trainer.evaluator.compute()
                logger.info(f"{Fore.YELLOW}  Epoch {epoch}/{self.eval_epochs} | Acc: {metrics['acc']:.4f}")
        
        # Final
        trainer.model.eval()
        for batch in trainer.test_dataloader:
            batch.pop('vids')
            batch.pop('neighbor_vids')
            inputs = {k: v.to(trainer.device) for k, v in batch.items()}
            labels = inputs.pop('labels')
            with torch.no_grad():
                outputs = trainer.model(**inputs)
                text_p = self.text_l2p(outputs['text_feat'])
                vision_p = self.vision_l2p(outputs['vision_feat'])
                audio_p = self.audio_l2p(outputs['audio_feat'])
                l2p_out = trainer.model.umaf(text_p, vision_p, audio_p)
                _, preds = torch.max(l2p_out['logits'], 1)
                trainer.evaluator.update(preds, labels)
        final_metrics = trainer.evaluator.compute()
        acc = final_metrics['acc']
        logger.info(f"{Fore.GREEN}✓ Final Acc: {acc:.4f}\n")
        return acc
    
    def run(self):
        """Run full CL experiment"""
        base_dir = Path.cwd()
        script_dir = base_dir / 'src' if (base_dir / 'src').exists() else base_dir
        
        tasks = [
            {'id': 0, 'name': 'HateMM', 'script': str(script_dir / 'CL_based_recent_short.py'), 'config': '84+'},
            {'id': 1, 'name': 'MHClip-EN', 'script': str(script_dir / 'CL_Multiclip_en.py'), 'config': '84+'},
            {'id': 2, 'name': 'MHClip-ZH', 'script': str(script_dir / 'CL_Multiclip_zn.py'), 'config': '84+'},
        ]
        
        for current_idx, task in enumerate(tasks):
            logger.info(f"{Fore.MAGENTA}{'='*80}")
            logger.info(f"{Fore.MAGENTA}PHASE {current_idx + 1}: Training {task['name']}")
            logger.info(f"{Fore.MAGENTA}{'='*80}\n")
            
            self.train_task(task['id'], task['name'], task['script'], task['config'])
            
            current_row = []
            for eval_idx in range(current_idx + 1):
                eval_task = tasks[eval_idx]
                acc = self.evaluate_task(eval_task['id'], eval_task['name'], eval_task['script'], eval_task['config'])
                current_row.append(acc)
            
            self.accuracy_matrix.append(current_row)
        
        self.print_results(tasks)
    
    def print_results(self, tasks):
        """Print final results"""
        logger.info(f"\n{Fore.MAGENTA}{'='*80}")
        logger.info(f"{Fore.MAGENTA}FINAL RESULTS - Learning to Prompt (L2P)")
        logger.info(f"{Fore.MAGENTA}{'='*80}\n")
        
        final_accs = self.accuracy_matrix[-1]
        avg_acc = np.mean(final_accs)
        
        forgetting = []
        for i in range(len(self.accuracy_matrix) - 1):
            forgetting.append(max(0, self.accuracy_matrix[i][i] - final_accs[i]))
        avg_forget = np.mean(forgetting) if forgetting else 0.0
        
        logger.info(f"{Fore.CYAN}Accuracy Matrix:")
        for i, row in enumerate(self.accuracy_matrix):
            logger.info(f"{Fore.CYAN}  {tasks[i]['name']}: {[f'{x:.3f}' for x in row]}")
        
        logger.info(f"\n{Fore.GREEN}Final Accuracies:")
        logger.info(f"{Fore.GREEN}  HateMM:     {final_accs[0]:.3f}")
        logger.info(f"{Fore.GREEN}  MHClip-EN:  {final_accs[1]:.3f}")
        logger.info(f"{Fore.GREEN}  MHClip-ZH:  {final_accs[2]:.3f}")
        logger.info(f"{Fore.GREEN}Average Accuracy:  {avg_acc:.3f} ↑")
        logger.info(f"{Fore.GREEN}Average Forgetting: {avg_forget:.3f} ↓")
        
        logger.info(f"\n{Fore.YELLOW}TABLE FORMAT:")
        logger.info(f"{Fore.YELLOW}L2P | {final_accs[0]:.3f} | {final_accs[1]:.3f} | {final_accs[2]:.3f} | {avg_acc:.3f} | {avg_forget:.3f}")
        
        # Save
        import json
        results = {
            'method': 'L2P',
            'accuracy_matrix': [[float(x) for x in row] for row in self.accuracy_matrix],
            'final_accuracies': {'HateMM': float(final_accs[0]), 'MHClip-EN': float(final_accs[1]), 'MHClip-ZH': float(final_accs[2])},
            'avg_accuracy': float(avg_acc),
            'avg_forgetting': float(avg_forget)
        }
        with open(self.save_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n{Fore.GREEN}✓ Results saved to: {self.save_path / 'results.json'}")


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    colorama.init()
    
    experiment = L2P_ContinualLearning(seed=2024, eval_epochs=40)
    experiment.run()
    
    logger.info(f"\n{Fore.CYAN}{'='*80}")
    logger.info(f"{Fore.CYAN}L2P EXPERIMENT COMPLETED")
    logger.info(f"{Fore.CYAN}{'='*80}\n")


if __name__ == '__main__':
    main()