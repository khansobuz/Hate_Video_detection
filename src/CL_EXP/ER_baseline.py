"""
BASELINE 1: Experience Replay (ER)
Complete implementation - just copy and run!

Usage:
    python ER_baseline.py

This will train on HateMM → MHClip-EN → MHClip-ZH with Experience Replay
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
# Experience Replay Buffer
# ============================================================================

class ERBuffer:
    """Experience Replay Buffer - stores samples from previous tasks"""
    def __init__(self, memory_size=1000, num_classes=2):
        self.memory_size = memory_size
        self.num_classes = num_classes
        self.per_class_size = memory_size // num_classes
        self.buffer = {c: {'feats': [], 'labels': []} for c in range(num_classes)}
    
    def add_samples(self, features, labels):
        """Add samples to buffer"""
        features = features.detach().cpu()
        for feat, label in zip(features, labels):
            label_int = label.item() if torch.is_tensor(label) else int(label)
            if len(self.buffer[label_int]['feats']) < self.per_class_size:
                self.buffer[label_int]['feats'].append(feat)
                self.buffer[label_int]['labels'].append(label_int)
            else:
                idx = np.random.randint(0, self.per_class_size)
                self.buffer[label_int]['feats'][idx] = feat
                self.buffer[label_int]['labels'][idx] = label_int
    
    def sample_batch(self, batch_size, device):
        """Sample from buffer"""
        if self.is_empty():
            return None, None
        samples_per_class = batch_size // self.num_classes
        all_feats, all_labels = [], []
        for c in range(self.num_classes):
            if len(self.buffer[c]['feats']) > 0:
                n = min(samples_per_class, len(self.buffer[c]['feats']))
                indices = np.random.choice(len(self.buffer[c]['feats']), n, replace=False)
                all_feats.extend([self.buffer[c]['feats'][i] for i in indices])
                all_labels.extend([self.buffer[c]['labels'][i] for i in indices])
        if not all_feats:
            return None, None
        return torch.stack(all_feats).to(device), torch.tensor(all_labels, dtype=torch.long).to(device)
    
    def is_empty(self):
        return all(len(self.buffer[c]['feats']) == 0 for c in range(self.num_classes))


# ============================================================================
# ER Continual Learning Experiment
# ============================================================================

class ER_ContinualLearning:
    def __init__(self, seed=2024, eval_epochs=40):
        self.seed = seed
        self.eval_epochs = eval_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(seed)
        
        self.accuracy_matrix = []
        self.save_path = Path(f'ER_results_{datetime.now().strftime("%m%d-%H%M%S")}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # ER buffer
        self.replay_buffer = ERBuffer(memory_size=1000, num_classes=2)
        self.replay_ratio = 0.3
        
        logger.info(f"{Fore.CYAN}{'='*80}")
        logger.info(f"{Fore.CYAN}Experience Replay (ER) Baseline")
        logger.info(f"{Fore.CYAN}Memory Size: 1000, Replay Ratio: 0.3")
        logger.info(f"{Fore.CYAN}{'='*80}\n")
    
    def train_task(self, task_id, task_name, script_name, config_name):
        """Train on a task with ER"""
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
        
        # Patch training to add ER
        original_train = trainer._train
        def train_with_er(epoch):
            loss_list = []
            trainer.model.train()
            for batch in tqdm(trainer.train_dataloader, desc=f'Train E{epoch}'):
                vids = batch.pop('vids')
                batch.pop('neighbor_vids')
                inputs = {k: v.to(trainer.device) for k, v in batch.items()}
                labels = inputs.pop('labels')
                
                # Forward
                outputs = trainer.model(**inputs)
                loss, _ = trainer.model.calculate_loss(**outputs, label=labels, epoch=epoch)
                
                # Add ER replay loss
                if not self.replay_buffer.is_empty():
                    replay_size = max(1, int(len(labels) * self.replay_ratio))
                    replay_feats, replay_labels = self.replay_buffer.sample_batch(replay_size, trainer.device)
                    if replay_feats is not None:
                        replay_out = trainer.model.umaf(replay_feats, replay_feats, replay_feats)
                        replay_loss = F.cross_entropy(replay_out['logits'], replay_labels)
                        loss = loss + 0.5 * replay_loss
                
                # Update buffer
                combined = (outputs['text_feat'] + outputs['vision_feat'] + outputs['audio_feat']) / 3.0
                self.replay_buffer.add_samples(combined, labels)
                
                # Backward
                _, preds = torch.max(outputs['logits'], 1)
                trainer.evaluator.update(preds, labels)
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=0.5)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            
            metrics = trainer.evaluator.compute()
            logger.info(f"{Fore.BLUE}E{epoch} Train | Loss: {np.mean(loss_list):.4f} | Acc: {metrics['acc']:.4f}")
        
        trainer._train = train_with_er
        
        # Train
        trainer.run()
        
        # Save
        model_path = self.save_path / f'model_task_{task_id}.pth'
        torch.save(trainer.model.state_dict(), model_path)
        self.previous_model_path = model_path
        
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
        
        # Fine-tune
        for epoch in range(self.eval_epochs):
            trainer._train(epoch=epoch)
            if epoch % 10 == 0:
                metrics = trainer._valid(split='test', epoch=epoch, final=False)
                logger.info(f"{Fore.YELLOW}  Epoch {epoch}/{self.eval_epochs} | Acc: {metrics['acc']:.4f}")
        
        # Final eval
        trainer.model.eval()
        final_metrics = trainer._valid(split='test', epoch=self.eval_epochs-1, final=True)
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
        logger.info(f"{Fore.MAGENTA}FINAL RESULTS - Experience Replay (ER)")
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
        logger.info(f"{Fore.YELLOW}ER | {final_accs[0]:.3f} | {final_accs[1]:.3f} | {final_accs[2]:.3f} | {avg_acc:.3f} | {avg_forget:.3f}")
        
        # Save
        import json
        results = {
            'method': 'ER',
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
    
    experiment = ER_ContinualLearning(seed=2024, eval_epochs=40)
    experiment.run()
    
    logger.info(f"\n{Fore.CYAN}{'='*80}")
    logger.info(f"{Fore.CYAN}ER EXPERIMENT COMPLETED")
    logger.info(f"{Fore.CYAN}{'='*80}\n")


if __name__ == '__main__':
    main()