"""
Continual Learning Experiment: HateMM → MHClip-EN → MHClip-ZH
With 40-epoch fine-tuning during evaluation for better continual learning
Computes Average Accuracy and Average Forgetting for CL-ReGAF method
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from loguru import logger
import colorama
from colorama import Fore
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


class ContinualLearningEvaluator:
    """Evaluates continual learning with 40-epoch fine-tuning"""
    
    def __init__(self, seed=2024, eval_epochs=40):
        self.seed = seed
        self.eval_epochs = eval_epochs  # Number of epochs for evaluation/fine-tuning
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {
            'task_accuracies': {},
            'final_accuracies': {},
            'forgetting': {}
        }
        self.accuracy_matrix = []
        self.save_path = Path(f'continual_learning_results/{datetime.now().strftime("%m%d-%H%M%S")}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{Fore.CYAN}Continual Learning Setup:")
        logger.info(f"{Fore.CYAN}  Training: Full epochs from config")
        logger.info(f"{Fore.CYAN}  Evaluation: {eval_epochs} epochs fine-tuning")
        
    def train_task(self, task_id, task_name, script_name, config_name):
        """Train on a specific task"""
        logger.info(f"{Fore.CYAN}{'='*80}")
        logger.info(f"{Fore.CYAN}Training Task {task_id + 1}: {task_name}")
        logger.info(f"{Fore.CYAN}{'='*80}")
        
        import importlib.util
        from omegaconf import OmegaConf
        
        if not Path(script_name).exists():
            raise FileNotFoundError(f"Script not found: {script_name}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(f"task_{task_id}_module", script_name)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        # Find config
        base_dir = Path.cwd()
        possible_config_paths = [
            base_dir / "src" / "config" / f"{config_name}.yaml",
            base_dir / "config" / f"{config_name}.yaml",
            Path("config") / f"{config_name}.yaml"
        ]
        
        config_path = None
        for path in possible_config_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Config not found: {config_name}.yaml")
        
        cfg = OmegaConf.load(config_path)
        cfg.seed = self.seed
        
        # Initialize trainer
        trainer = task_module.Trainer(cfg)
        
        # Store reference for loading
        if hasattr(self, 'previous_model_path'):
            previous_path = self.previous_model_path
        else:
            previous_path = None
        
        # Patch _reset to load previous model weights
        original_reset = trainer._reset
        
        def patched_reset(cfg, fold, type):
            result = original_reset(cfg, fold, type)
            
            # Load previous weights if this is not the first task
            if previous_path is not None and Path(previous_path).exists():
                logger.info(f"{Fore.YELLOW}Loading previous model weights...")
                state_dict = torch.load(previous_path, weights_only=False)
                trainer.model.load_state_dict(state_dict, strict=False)
                logger.info(f"{Fore.GREEN}✓ Loaded previous weights")
            
            return result
        
        trainer._reset = patched_reset
        
        # Train
        trainer.run()
        
        # Save model
        model_save_path = self.save_path / f'model_after_task_{task_id}.pth'
        torch.save(trainer.model.state_dict(), model_save_path)
        self.previous_model_path = model_save_path
        
        logger.info(f"{Fore.GREEN}✓ Task {task_id + 1} training completed")
        logger.info(f"{Fore.GREEN}Model saved to: {model_save_path}")
        
        return trainer
    
    def evaluate_task_with_finetuning(self, task_id, task_name, script_name, config_name):
        """
        Evaluate on a specific task with fine-tuning
        This loads the current model and fine-tunes it for eval_epochs on the evaluation task
        """
        logger.info(f"{Fore.YELLOW}{'='*80}")
        logger.info(f"{Fore.YELLOW}Evaluating on Task {task_id + 1}: {task_name}")
        logger.info(f"{Fore.YELLOW}Fine-tuning for {self.eval_epochs} epochs")
        logger.info(f"{Fore.YELLOW}{'='*80}")
        
        import importlib.util
        from omegaconf import OmegaConf
        
        # Load module with unique name
        spec = importlib.util.spec_from_file_location(f"eval_{task_id}_{datetime.now().strftime('%H%M%S%f')}", script_name)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        # Find config
        base_dir = Path.cwd()
        possible_config_paths = [
            base_dir / "src" / "config" / f"{config_name}.yaml",
            base_dir / "config" / f"{config_name}.yaml",
            Path("config") / f"{config_name}.yaml"
        ]
        
        config_path = None
        for path in possible_config_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Config not found: {config_name}.yaml")
        
        cfg = OmegaConf.load(config_path)
        cfg.seed = self.seed
        
        # Override number of epochs for evaluation
        original_num_epochs = cfg.num_epoch
        cfg.num_epoch = self.eval_epochs
        
        # Initialize trainer
        trainer = task_module.Trainer(cfg)
        
        # Patch _reset to load current model state
        original_reset = trainer._reset
        model_loaded = False
        
        def patched_reset_eval(cfg, fold, type):
            nonlocal model_loaded
            result = original_reset(cfg, fold, type)
            
            # Load current model state from continual learning
            if hasattr(self, 'previous_model_path') and not model_loaded:
                logger.info(f"{Fore.YELLOW}Loading current model state for fine-tuning...")
                state_dict = torch.load(self.previous_model_path, weights_only=False)
                trainer.model.load_state_dict(state_dict, strict=False)
                model_loaded = True
                logger.info(f"{Fore.GREEN}✓ Loaded current model state")
            
            return result
        
        trainer._reset = patched_reset_eval
        
        # Fine-tune for eval_epochs
        logger.info(f"{Fore.YELLOW}Starting {self.eval_epochs}-epoch fine-tuning...")
        
        # Call _reset to initialize
        trainer._reset(cfg, fold='default', type=cfg.type)
        
        # Fine-tune by running limited epochs
        for epoch in range(self.eval_epochs):
            trainer._train(epoch=epoch)
            
            # Validate every 10 epochs
            if epoch % 10 == 0 or epoch == self.eval_epochs - 1:
                metrics = trainer._valid(split='test', epoch=epoch, use_earlystop=False, final=False)
                logger.info(f"{Fore.YELLOW}Epoch {epoch}/{self.eval_epochs} | Test Acc: {metrics['acc']:.5f} | F1: {metrics['macro_f1']:.5f}")
        
        # Final evaluation
        trainer.model.eval()
        final_metrics = trainer._valid(split='test', epoch=self.eval_epochs-1, use_earlystop=False, final=True)
        
        accuracy = final_metrics['acc']
        logger.info(f"{Fore.GREEN}✓ Task {task_id + 1} Final Accuracy after fine-tuning: {accuracy:.5f}")
        
        return accuracy
    
    def run_continual_learning(self):
        """Main continual learning loop"""
        
        # Auto-detect script directory
        base_dir = Path.cwd()
        if (base_dir / 'src' / 'CL_based_recent_short.py').exists():
            script_dir = base_dir / 'src'
            logger.info(f"{Fore.CYAN}Found scripts in: {script_dir}")
        else:
            script_dir = base_dir
            logger.info(f"{Fore.CYAN}Using current directory: {script_dir}")
        
        tasks = [
            {
                'id': 0,
                'name': 'HateMM',
                'script': str(script_dir / 'CL_based_recent_short.py'),
                'config': '84+'
            },
            {
                'id': 1,
                'name': 'MHClip-EN',
                'script': str(script_dir / 'CL_Multiclip_en.py'),
                'config': '84+'
            },
            {
                'id': 2,
                'name': 'MHClip-ZH',
                'script': str(script_dir / 'CL_Multiclip_zn.py'),
                'config': '84+'
            }
        ]
        
        # Check files
        logger.info(f"{Fore.CYAN}Checking required files...")
        for task in tasks:
            if not Path(task['script']).exists():
                raise FileNotFoundError(f"Script not found: {task['script']}")
            logger.info(f"  ✓ Found: {task['script']}")
        
        accuracy_matrix = []
        
        for current_task_idx, task in enumerate(tasks):
            # Train on current task
            logger.info(f"{Fore.MAGENTA}{'='*80}")
            logger.info(f"{Fore.MAGENTA}PHASE {current_task_idx + 1}: Training on {task['name']}")
            logger.info(f"{Fore.MAGENTA}{'='*80}")
            
            trainer = self.train_task(
                task['id'],
                task['name'],
                task['script'],
                task['config']
            )
            
            # Evaluate on all seen tasks WITH fine-tuning
            logger.info(f"{Fore.MAGENTA}{'='*80}")
            logger.info(f"{Fore.MAGENTA}Evaluating on all {current_task_idx + 1} seen tasks")
            logger.info(f"{Fore.MAGENTA}{'='*80}")
            
            current_row = []
            
            for eval_task_idx in range(current_task_idx + 1):
                eval_task = tasks[eval_task_idx]
                
                # Fine-tune and evaluate
                acc = self.evaluate_task_with_finetuning(
                    eval_task['id'],
                    eval_task['name'],
                    eval_task['script'],
                    eval_task['config']
                )
                
                current_row.append(acc)
                
                key = f"after_task_{current_task_idx}_eval_task_{eval_task_idx}"
                self.results['task_accuracies'][key] = {
                    'trained_on': task['name'],
                    'evaluated_on': eval_task['name'],
                    'accuracy': acc
                }
            
            accuracy_matrix.append(current_row)
            
            logger.info(f"{Fore.CYAN}{'='*80}")
            logger.info(f"{Fore.CYAN}Results after training on {task['name']}:")
            for i, acc in enumerate(current_row):
                logger.info(f"{Fore.CYAN}  Task {i + 1} ({tasks[i]['name']}): {acc:.5f}")
            logger.info(f"{Fore.CYAN}{'='*80}")
        
        self.accuracy_matrix = accuracy_matrix
        self.compute_metrics(tasks)
        self.save_results()
    
    def compute_metrics(self, tasks):
        """Compute Average Accuracy and Average Forgetting"""
        
        n_tasks = len(self.accuracy_matrix)
        final_accuracies = self.accuracy_matrix[-1]
        
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        logger.info(f"{Fore.MAGENTA}FINAL CONTINUAL LEARNING RESULTS")
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        
        # Accuracy matrix
        logger.info(f"{Fore.CYAN}Accuracy Matrix (with {self.eval_epochs}-epoch fine-tuning):")
        logger.info(f"{Fore.CYAN}{'Task Trained':<15} | " + " | ".join([f"{tasks[i]['name']:<12}" for i in range(n_tasks)]))
        logger.info(f"{Fore.CYAN}{'-'*80}")
        
        for i in range(n_tasks):
            row_str = f"{tasks[i]['name']:<15} | "
            for j in range(len(self.accuracy_matrix[i])):
                row_str += f"{self.accuracy_matrix[i][j]:.5f}      | "
            logger.info(f"{Fore.CYAN}{row_str}")
        
        # Average Accuracy
        avg_accuracy = np.mean(final_accuracies)
        
        # Average Forgetting
        forgetting_values = []
        
        for task_idx in range(n_tasks - 1):
            max_acc = self.accuracy_matrix[task_idx][task_idx]
            final_acc = final_accuracies[task_idx]
            forgetting = max(0, max_acc - final_acc)
            forgetting_values.append(forgetting)
            
            logger.info(f"{Fore.YELLOW}Task {task_idx + 1} ({tasks[task_idx]['name']}):")
            logger.info(f"  Max Acc: {max_acc:.5f} (after initial training)")
            logger.info(f"  Final Acc: {final_acc:.5f} (after fine-tuning on all tasks)")
            logger.info(f"  Forgetting: {forgetting:.5f}")
        
        avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
        
        # Store results
        self.results['final_accuracies'] = {
            tasks[i]['name']: final_accuracies[i] for i in range(n_tasks)
        }
        self.results['average_accuracy'] = avg_accuracy
        self.results['average_forgetting'] = avg_forgetting
        self.results['forgetting_per_task'] = {
            tasks[i]['name']: forgetting_values[i] for i in range(len(forgetting_values))
        }
        self.results['eval_epochs'] = self.eval_epochs
        
        # Summary
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        logger.info(f"{Fore.MAGENTA}SUMMARY")
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        logger.info(f"{Fore.GREEN}Method: CL-ReGAF (Your Method)")
        logger.info(f"{Fore.GREEN}Evaluation: {self.eval_epochs}-epoch fine-tuning per task")
        logger.info(f"{Fore.GREEN}Final Accuracies:")
        for i in range(n_tasks):
            logger.info(f"  {tasks[i]['name']:<12}: {final_accuracies[i]:.3f}")
        logger.info(f"{Fore.GREEN}Average Accuracy: {avg_accuracy:.3f} ↑")
        logger.info(f"{Fore.GREEN}Average Forgetting: {avg_forgetting:.3f} ↓")
        
        # Table format
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        logger.info(f"{Fore.MAGENTA}TABLE FORMAT:")
        logger.info(f"{Fore.MAGENTA}{'='*80}")
        table_str = f"CL-ReGAF | "
        for i in range(n_tasks):
            table_str += f"{final_accuracies[i]:.3f} | "
        table_str += f"{avg_accuracy:.3f} | {avg_forgetting:.3f}"
        logger.info(f"{Fore.GREEN}{table_str}")
        
        return avg_accuracy, avg_forgetting
    
    def save_results(self):
        """Save results to JSON and CSV"""
        results_file = self.save_path / 'continual_learning_results.json'
        
        results_to_save = {
            'eval_epochs': self.eval_epochs,
            'accuracy_matrix': [[float(x) for x in row] for row in self.accuracy_matrix],
            'final_accuracies': {k: float(v) for k, v in self.results['final_accuracies'].items()},
            'average_accuracy': float(self.results['average_accuracy']),
            'average_forgetting': float(self.results['average_forgetting']),
            'forgetting_per_task': {k: float(v) for k, v in self.results['forgetting_per_task'].items()},
            'task_accuracies': {k: {
                'trained_on': v['trained_on'],
                'evaluated_on': v['evaluated_on'],
                'accuracy': float(v['accuracy'])
            } for k, v in self.results['task_accuracies'].items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"{Fore.GREEN}✓ Results saved to: {results_file}")
        
        # CSV - handle variable row lengths
        csv_file = self.save_path / 'accuracy_matrix.csv'
        tasks = ['HateMM', 'MHClip-EN', 'MHClip-ZH']
        
        # Pad rows to make them equal length
        max_len = max(len(row) for row in self.accuracy_matrix)
        padded_matrix = []
        for row in self.accuracy_matrix:
            padded_row = row + [np.nan] * (max_len - len(row))
            padded_matrix.append(padded_row)
        
        df = pd.DataFrame(padded_matrix, 
                         columns=tasks[:max_len], 
                         index=tasks[:len(self.accuracy_matrix)])
        df.to_csv(csv_file)
        logger.info(f"{Fore.GREEN}✓ Accuracy matrix saved to: {csv_file}")


def main():
    """Main entry point"""
    
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info(f"{Fore.CYAN}{'='*80}")
    logger.info(f"{Fore.CYAN}CONTINUAL LEARNING EXPERIMENT WITH FINE-TUNING")
    logger.info(f"{Fore.CYAN}Sequential Training: HateMM → MHClip-EN → MHClip-ZH")
    logger.info(f"{Fore.CYAN}{'='*80}")
    
    colorama.init()
    
    seed = 2024
    set_seed(seed)
    
    # Initialize evaluator with 40 epochs for evaluation/fine-tuning
    evaluator = ContinualLearningEvaluator(seed=seed, eval_epochs=40)
    
    try:
        evaluator.run_continual_learning()
    except Exception as e:
        logger.error(f"{Fore.RED}Error during continual learning: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{Fore.CYAN}{'='*80}")
    logger.info(f"{Fore.CYAN}EXPERIMENT COMPLETED")
    logger.info(f"{Fore.CYAN}{'='*80}")


if __name__ == '__main__':
    main()