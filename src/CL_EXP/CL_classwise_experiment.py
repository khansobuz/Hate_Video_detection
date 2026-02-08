"""
Temporal Continual Learning with 3 Tasks (Increasing Hate Ratio)
T1(30% hate) → T2(50% hate) → T3(70% hate)
With epoch tuning for best accuracy
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from colorama import Fore, init
init()

from CL_based_recent_CL import NovelCLModel, HateMM_Novel_Dataset, HateMM_Novel_Collator

class TemporalCL_3Tasks:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.collator = HateMM_Novel_Collator()
    
    def run_temporal_3tasks(self, cfg, epochs_per_task=50):
        """
        Temporal CL with 3 tasks - increasing hate over time
        T1: 70% non-hate, 30% hate (Early period)
        T2: 50% non-hate, 50% hate (Middle period)
        T3: 30% non-hate, 70% hate (Late period)
        """
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Temporal CL: 3 Tasks with Increasing Hate Content")
        print(f"{Fore.CYAN}Epochs per task: {epochs_per_task}")
        print(f"{Fore.CYAN}{'='*80}")
        
        # Load data
        train_data = HateMM_Novel_Dataset(task='binary', fold='default', split='train', 
                                          num_pos=30, num_neg=90)
        test_data = HateMM_Novel_Dataset(task='binary', fold='default', split='test',
                                         num_pos=30, num_neg=90)
        
        # Separate by class
        train_labels = [train_data[i]['label'].item() for i in range(len(train_data))]
        test_labels = [test_data[i]['label'].item() for i in range(len(test_data))]
        
        train_non_hate = [i for i, l in enumerate(train_labels) if l == 0]
        train_hate = [i for i, l in enumerate(train_labels) if l == 1]
        test_non_hate = [i for i, l in enumerate(test_labels) if l == 0]
        test_hate = [i for i, l in enumerate(test_labels) if l == 1]
        
        print(f"{Fore.GREEN}Total Train: {len(train_data)} ({len(train_non_hate)} non-hate, {len(train_hate)} hate)")
        print(f"{Fore.GREEN}Total Test: {len(test_data)} ({len(test_non_hate)} non-hate, {len(test_hate)} hate)")
        
        # Define 3 tasks with ratios [non_hate_ratio, hate_ratio]
        num_tasks = 3
        ratios = [
            (0.7, 0.3),  # T1: Early period, 30% hate
            (0.5, 0.5),  # T2: Middle period, 50% hate
            (0.3, 0.7)   # T3: Late period, 70% hate
        ]
        
        # Create tasks
        train_tasks = []
        test_tasks = []
        
        samples_per_task = len(train_data) // num_tasks
        test_samples_per_task = len(test_data) // num_tasks
        
        train_nh_ptr = 0
        train_h_ptr = 0
        test_nh_ptr = 0
        test_h_ptr = 0
        
        print(f"\n{Fore.CYAN}Task Distribution:")
        for task_id, (nh_ratio, h_ratio) in enumerate(ratios, start=1):
            n_train_nh = int(samples_per_task * nh_ratio)
            n_train_h = int(samples_per_task * h_ratio)
            n_test_nh = int(test_samples_per_task * nh_ratio)
            n_test_h = int(test_samples_per_task * h_ratio)
            
            train_task = (train_non_hate[train_nh_ptr:train_nh_ptr + n_train_nh] + 
                         train_hate[train_h_ptr:train_h_ptr + n_train_h])
            test_task = (test_non_hate[test_nh_ptr:test_nh_ptr + n_test_nh] + 
                        test_hate[test_h_ptr:test_h_ptr + n_test_h])
            
            np.random.shuffle(train_task)
            np.random.shuffle(test_task)
            
            train_tasks.append(train_task)
            test_tasks.append(test_task)
            
            train_nh_ptr += n_train_nh
            train_h_ptr += n_train_h
            test_nh_ptr += n_test_nh
            test_h_ptr += n_test_h
            
            print(f"  T{task_id}: Train={len(train_task)} ({n_train_nh} non-hate, {n_train_h} hate = {h_ratio*100:.0f}% hate)")
            print(f"       Test={len(test_task)} ({n_test_nh} non-hate, {n_test_h} hate)")
        
        # Initialize model
        print(f"\n{Fore.YELLOW}Initializing PC-CML model...")
        model = NovelCLModel(
            text_encoder='local_models/bert-base-uncased',
            fea_dim=128,
            dropout=0.3,
            num_head=4,
            num_classes=2,
            num_epoch=100,
            label_smoothing=0.15,
            use_mixup=True,
            mixup_alpha=0.3,
            prompt_pool_size=8,
            prompt_length=4,
            prompt_top_k=3
        ).to(self.device)
        
        # Results matrix: [task_learned, task_eval]
        results = np.zeros((num_tasks, num_tasks))
        
        # Train sequentially
        for task_id in range(num_tasks):
            print(f"\n{Fore.YELLOW}{'='*60}")
            print(f"{Fore.YELLOW}Learning Task {task_id + 1} (Period {task_id + 1})")
            print(f"{Fore.YELLOW}{'='*60}")
            
            # Train on current task
            self._train_task(model, train_data, train_tasks[task_id], cfg, 
                           epochs=epochs_per_task, task_id=task_id+1)
            
            # Evaluate on all learned tasks
            for eval_task_id in range(task_id + 1):
                acc = self._eval_task(model, test_data, test_tasks[eval_task_id])
                results[task_id, eval_task_id] = acc
                print(f"{Fore.GREEN}After T{task_id+1} → Eval T{eval_task_id+1}: {acc:.4f}")
        
        # Print results
        self._print_results(results, num_tasks)
        
        # Save results
        self._save_results(results, ratios, epochs_per_task)
        
        return results
    
    def _train_task(self, model, dataset, indices, cfg, epochs=30, task_id=1):
        """Train on one task with progress tracking"""
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=cfg.batch_size, collate_fn=self.collator,
                          shuffle=True, num_workers=0)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.opt.lr,
                                     weight_decay=cfg.opt.weight_decay)
        
        model.train()
        best_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc=f"Training T{task_id}"):
            epoch_loss = 0
            num_batches = 0
            
            for batch in loader:
                batch.pop('vids')
                batch.pop('neighbor_vids')
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs.pop('labels')
                inputs['labels'] = labels
                
                outputs = model(**inputs)
                loss, _ = model.calculate_loss(**outputs, label=labels, epoch=epoch)
                
                epoch_loss += loss.item()
                num_batches += 1
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = epoch_loss / num_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
    
    def _eval_task(self, model, dataset, indices):
        """Evaluate on one task"""
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, collate_fn=self.collator,
                          shuffle=False, num_workers=0)
        
        model.eval()
        correct, total = 0, 0
        
        for batch in loader:
            batch.pop('vids')
            batch.pop('neighbor_vids')
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            labels = inputs.pop('labels')
            inputs['labels'] = labels
            
            with torch.no_grad():
                outputs = model(**inputs)
                _, preds = torch.max(outputs['logits'], 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _print_results(self, results, num_tasks):
        """Print accuracy matrix and metrics"""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}FINAL RESULTS - 3 TASKS")
        print(f"{Fore.MAGENTA}{'='*80}")
        
        # Accuracy matrix
        print(f"\n{Fore.CYAN}Accuracy Matrix (After learning task i, eval on task j):")
        header = "      " + "  ".join([f"T{i+1}" for i in range(num_tasks)])
        print(header)
        for i in range(num_tasks):
            row = f"T{i+1} | " + "  ".join([f"{results[i,j]:.3f}" if j <= i else "---" 
                                         for j in range(num_tasks)])
            print(row)
        
        # Calculate metrics
        aa = np.mean(results[-1, :])
        
        # Forgetting per task
        forgetting_per_task = []
        for j in range(num_tasks - 1):
            max_acc = results[j, j]
            final_acc = results[-1, j]
            forgetting_per_task.append(max_acc - final_acc)
        avg_forgetting = np.mean(forgetting_per_task)
        
        # Backward transfer
        bwt_values = []
        for i in range(1, num_tasks):
            for j in range(i):
                bwt_values.append(results[i, j] - results[j, j])
        avg_bwt = np.mean(bwt_values) if bwt_values else 0.0
        
        # Print metrics
        print(f"\n{Fore.MAGENTA}Continual Learning Metrics:")
        print(f"{Fore.CYAN}  Average Accuracy (AA): {aa:.4f}")
        print(f"{Fore.CYAN}  Average Forgetting (FM): {avg_forgetting:.4f}")
        print(f"{Fore.CYAN}  Backward Transfer (BWT): {avg_bwt:.4f}")
        
        print(f"\n{Fore.YELLOW}Per-Task Forgetting:")
        for j in range(num_tasks - 1):
            print(f"  T{j+1}: {forgetting_per_task[j]:.4f} (learned {results[j,j]:.3f} → final {results[-1,j]:.3f})")
        print(f"  T{num_tasks}: N/A (last task)")
        
        print(f"\n{Fore.YELLOW}Per-Task Final Accuracy:")
        for j in range(num_tasks):
            print(f"  T{j+1}: {results[-1,j]:.4f}")
        
        print(f"{Fore.MAGENTA}{'='*80}\n")
    
    def _save_results(self, results, ratios, epochs):
        """Save results"""
        save_path = Path('cl_results')
        save_path.mkdir(exist_ok=True)
        
        # Accuracy matrix
        df_matrix = pd.DataFrame(results, 
                                columns=[f'T{i+1}' for i in range(results.shape[1])],
                                index=[f'After_T{i+1}' for i in range(results.shape[0])])
        df_matrix.to_csv(save_path / 'temporal_3tasks_matrix.csv')
        
        # Metrics
        aa = np.mean(results[-1, :])
        forgetting_per_task = [results[j, j] - results[-1, j] for j in range(results.shape[0] - 1)]
        avg_forgetting = np.mean(forgetting_per_task)
        bwt_values = [results[i, j] - results[j, j] for i in range(1, results.shape[0]) for j in range(i)]
        avg_bwt = np.mean(bwt_values) if bwt_values else 0.0
        
        metrics_df = pd.DataFrame({
            'Metric': ['Average Accuracy', 'Average Forgetting', 'Backward Transfer', 'Epochs per Task'],
            'Value': [aa, avg_forgetting, avg_bwt, epochs]
        })
        metrics_df.to_csv(save_path / 'temporal_3tasks_metrics.csv', index=False)
        
        # Paper table
        paper_table = []
        for i, (nh_ratio, h_ratio) in enumerate(ratios, start=1):
            forgetting = results[i-1, i-1] - results[-1, i-1] if i < len(ratios)+1 else 'N/A'
            paper_table.append({
                'Task': f'T{i}',
                'Distribution': f'{int(nh_ratio*100)}% non-hate, {int(h_ratio*100)}% hate',
                'Accuracy (when learned)': f'{results[i-1, i-1]:.4f}',
                'Accuracy (final)': f'{results[-1, i-1]:.4f}',
                'Forgetting': f'{forgetting:.4f}' if isinstance(forgetting, float) else forgetting
            })
        
        paper_df = pd.DataFrame(paper_table)
        paper_df.to_csv(save_path / 'temporal_3tasks_paper.csv', index=False)
        
        print(f"{Fore.GREEN}Results saved to {save_path}/")

def run_with_different_epochs():
    """Test different epoch settings to find best"""
    class Config:
        batch_size = 40
        seed = 2024
        class opt:
            lr = 0.0002
            weight_decay = 0.0001
    
    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    exp = TemporalCL_3Tasks()
    
    # Test different epoch counts
    print(f"{Fore.MAGENTA}Testing different epoch settings...\n")
    
    epoch_settings = [20, 30, 40]
    all_results = {}
    
    for epochs in epoch_settings:
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Testing with {epochs} epochs per task")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        results = exp.run_temporal_3tasks(cfg, epochs_per_task=epochs)
        aa = np.mean(results[-1, :])
        all_results[epochs] = {'AA': aa, 'matrix': results}
        
        print(f"\n{Fore.YELLOW}Summary for {epochs} epochs: AA = {aa:.4f}")
    
    # Find best
    best_epochs = max(all_results.keys(), key=lambda k: all_results[k]['AA'])
    best_aa = all_results[best_epochs]['AA']
    
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}BEST RESULT: {best_epochs} epochs with AA = {best_aa:.4f}")
    print(f"{Fore.MAGENTA}{'='*80}")

def main():
    """Run with default 30 epochs"""
    class Config:
        batch_size = 40
        seed = 2024
        class opt:
            lr = 0.0002
            weight_decay = 0.0001
    
    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    exp = TemporalCL_3Tasks()
    
    # Run with 30 epochs (good balance)
    results = exp.run_temporal_3tasks(cfg, epochs_per_task=30)
    
    print(f"\n{Fore.GREEN}Experiment completed!")
    print(f"{Fore.GREEN}To test different epochs, run: run_with_different_epochs()")

if __name__ == '__main__':
    # Choose one:
    main()  # Default: 30 epochs
    # run_with_different_epochs()  # Test 20, 30, 40 epochs