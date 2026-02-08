"""
Detailed Performance Measurement - Step-by-Step
Shows performance for each component/step in the pipeline

Usage:
    python measure_performance_detailed.py
"""

import sys
import time
import psutil
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import colorama
from colorama import Fore
import pandas as pd
from datetime import datetime


class DetailedPerformanceMeasure:
    """Measures performance for each step in the pipeline"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = []
        
        logger.info(f"{Fore.CYAN}{'='*100}")
        logger.info(f"{Fore.CYAN}DETAILED PERFORMANCE MEASUREMENT - STEP BY STEP")
        logger.info(f"{Fore.CYAN}Device: {self.device}")
        logger.info(f"{Fore.CYAN}{'='*100}\n")
    
    def measure_step_by_step(self, dataset_name, script_name, config_name):
        """Measure performance for each step"""
        logger.info(f"\n{Fore.YELLOW}{'='*100}")
        logger.info(f"{Fore.YELLOW}Dataset: {dataset_name}")
        logger.info(f"{Fore.YELLOW}{'='*100}\n")
        
        import importlib.util
        from omegaconf import OmegaConf
        
        # Load module
        spec = importlib.util.spec_from_file_location(f"perf_{dataset_name}", script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Load config
        base_dir = Path.cwd()
        for base_path in [base_dir / "src" / "config", base_dir / "config"]:
            config_path = base_path / f"{config_name}.yaml"
            if config_path.exists():
                break
        
        cfg = OmegaConf.load(config_path)
        cfg.seed = 2024
        cfg.num_epoch = 1
        
        # Initialize
        trainer = module.Trainer(cfg)
        trainer._reset(cfg, 'default', cfg.type)
        trainer.model.eval()
        
        # Get one batch for detailed measurement
        test_batch = next(iter(trainer.test_dataloader))
        vids = test_batch.pop('vids')
        test_batch.pop('neighbor_vids')
        inputs = {k: v.to(trainer.device) for k, v in test_batch.items()}
        labels = inputs.pop('labels')
        batch_size = len(labels)
        
        logger.info(f"{Fore.CYAN}Batch size: {batch_size}")
        logger.info(f"{Fore.CYAN}Measuring each step...\n")
        
        # Measure memory
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        step_times = {}
        step_memories = {}
        step_latencies = {}
        step_qps = {}
        
        with torch.no_grad():
            # =================================================================
            # STEP 1: FAME (Feature Adaptation)
            # =================================================================
            logger.info(f"{Fore.YELLOW}Step 1: FAME (Feature Adaptation)")
            
            # Memory before Step 1
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            mem_step1_start = process.memory_info().rss / 1024 / 1024
            
            step1_start = time.time()
            
            # Get input features (already extracted)
            text_fea = inputs['text_fea']
            vision_fea = inputs['vision_fea']
            audio_fea = inputs['audio_fea']
            
            # Project features
            text_proj_start = time.time()
            if text_fea.dim() == 2:
                text_fea = text_fea.unsqueeze(1)
            if audio_fea.dim() == 2:
                audio_fea = audio_fea.unsqueeze(1)
            
            text_pooled = text_fea.mean(dim=1)
            vision_pooled = vision_fea.mean(dim=1)
            audio_pooled = audio_fea.mean(dim=1)
            
            text_projected = trainer.model.text_proj(text_pooled)
            text_proj_time = time.time() - text_proj_start
            
            vision_proj_start = time.time()
            vision_projected = trainer.model.vision_proj(vision_pooled)
            vision_proj_time = time.time() - vision_proj_start
            
            audio_proj_start = time.time()
            audio_projected = trainer.model.audio_proj(audio_pooled)
            audio_proj_time = time.time() - audio_proj_start
            
            # FAME adaptation
            text_adapted = trainer.model.text_fame(text_projected)
            vision_adapted = trainer.model.vision_fame(vision_projected)
            audio_adapted = trainer.model.audio_fame(audio_projected)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step1_time = time.time() - step1_start
            
            # Memory after Step 1
            mem_step1_end = process.memory_info().rss / 1024 / 1024
            if torch.cuda.is_available():
                gpu_mem_step1 = torch.cuda.max_memory_allocated() / 1024 / 1024
                step1_memory = (mem_step1_end - mem_step1_start) + gpu_mem_step1
            else:
                step1_memory = mem_step1_end - mem_step1_start
            
            step_times['Step 1: FAME'] = step1_time
            step_memories['Step 1: FAME'] = step1_memory
            step_latencies['Step 1: FAME'] = step1_time / batch_size
            step_qps['Step 1: FAME'] = batch_size / step1_time if step1_time > 0 else 0
            
            logger.info(f"{Fore.GREEN}  Time: {step1_time:.4f}s | Latency: {step_latencies['Step 1: FAME']:.6f}s/sample | QPS: {step_qps['Step 1: FAME']:.2f} | Memory: {step1_memory:.2f}MB\n")
            
            # =================================================================
            # STEP 2: R³GC (Graph Attention)
            # =================================================================
            logger.info(f"{Fore.YELLOW}Step 2: R³GC (Graph Attention)")
            
            # Memory before Step 2
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            mem_step2_start = process.memory_info().rss / 1024 / 1024
            
            step2_start = time.time()
            
            # Process neighbor features
            text_sim_pos = trainer.model.text_proj(inputs['text_sim_pos_fea'].mean(dim=1))
            text_sim_neg = trainer.model.text_proj(inputs['text_sim_neg_fea'].mean(dim=1))
            text_sim_pos_adapted = trainer.model.text_fame(text_sim_pos)
            text_sim_neg_adapted = trainer.model.text_fame(text_sim_neg)
            
            vision_sim_pos = trainer.model.vision_proj(inputs['vision_sim_pos_fea'].mean(dim=[1,2]))
            vision_sim_neg = trainer.model.vision_proj(inputs['vision_sim_neg_fea'].mean(dim=[1,2]))
            vision_sim_pos_adapted = trainer.model.vision_fame(vision_sim_pos)
            vision_sim_neg_adapted = trainer.model.vision_fame(vision_sim_neg)
            
            audio_sim_pos = trainer.model.audio_proj(inputs['audio_sim_pos_fea'].mean(dim=1))
            audio_sim_neg = trainer.model.audio_proj(inputs['audio_sim_neg_fea'].mean(dim=1))
            audio_sim_pos_adapted = trainer.model.audio_fame(audio_sim_pos)
            audio_sim_neg_adapted = trainer.model.audio_fame(audio_sim_neg)
            
            # Graph convolution
            text_graph = trainer.model.text_graph(text_adapted, text_sim_pos_adapted, text_sim_neg_adapted)
            vision_graph = trainer.model.vision_graph(vision_adapted, vision_sim_pos_adapted, vision_sim_neg_adapted)
            audio_graph = trainer.model.audio_graph(audio_adapted, audio_sim_pos_adapted, audio_sim_neg_adapted)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step2_time = time.time() - step2_start
            
            # Memory after Step 2
            mem_step2_end = process.memory_info().rss / 1024 / 1024
            if torch.cuda.is_available():
                gpu_mem_step2 = torch.cuda.max_memory_allocated() / 1024 / 1024
                step2_memory = (mem_step2_end - mem_step2_start) + gpu_mem_step2
            else:
                step2_memory = mem_step2_end - mem_step2_start
            
            step_times['Step 2: R³GC'] = step2_time
            step_memories['Step 2: R³GC'] = step2_memory
            step_latencies['Step 2: R³GC'] = step2_time / batch_size
            step_qps['Step 2: R³GC'] = batch_size / step2_time if step2_time > 0 else 0
            
            logger.info(f"{Fore.GREEN}  Time: {step2_time:.4f}s | Latency: {step_latencies['Step 2: R³GC']:.6f}s/sample | QPS: {step_qps['Step 2: R³GC']:.2f} | Memory: {step2_memory:.2f}MB\n")
            
            # =================================================================
            # STEP 3: CMEL/PC-CML (Memory Learning)
            # =================================================================
            logger.info(f"{Fore.YELLOW}Step 3: CMEL/PC-CML (Memory Learning)")
            
            # Memory before Step 3
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            mem_step3_start = process.memory_info().rss / 1024 / 1024
            
            step3_start = time.time()
            
            # CMEL or PC-CML processing
            if hasattr(trainer.model, 'cmel'):
                text_cml = trainer.model.cmel(text_graph)
                vision_cml = trainer.model.cmel(vision_graph)
                audio_cml = trainer.model.cmel(audio_graph)
            elif hasattr(trainer.model, 'pc_cml'):
                text_cml_out = trainer.model.pc_cml(text_graph, labels, train=False)
                vision_cml_out = trainer.model.pc_cml(vision_graph, labels, train=False)
                audio_cml_out = trainer.model.pc_cml(audio_graph, labels, train=False)
                text_cml = text_cml_out['features']
                vision_cml = vision_cml_out['features']
                audio_cml = audio_cml_out['features']
            else:
                text_cml = text_graph
                vision_cml = vision_graph
                audio_cml = audio_graph
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step3_time = time.time() - step3_start
            
            # Memory after Step 3
            mem_step3_end = process.memory_info().rss / 1024 / 1024
            if torch.cuda.is_available():
                gpu_mem_step3 = torch.cuda.max_memory_allocated() / 1024 / 1024
                step3_memory = (mem_step3_end - mem_step3_start) + gpu_mem_step3
            else:
                step3_memory = mem_step3_end - mem_step3_start
            
            step_times['Step 3: CMEL/PC-CML'] = step3_time
            step_memories['Step 3: CMEL/PC-CML'] = step3_memory
            step_latencies['Step 3: CMEL/PC-CML'] = step3_time / batch_size
            step_qps['Step 3: CMEL/PC-CML'] = batch_size / step3_time if step3_time > 0 else 0
            
            logger.info(f"{Fore.GREEN}  Time: {step3_time:.4f}s | Latency: {step_latencies['Step 3: CMEL/PC-CML']:.6f}s/sample | QPS: {step_qps['Step 3: CMEL/PC-CML']:.2f} | Memory: {step3_memory:.2f}MB\n")
            
            # =================================================================
            # STEP 4: UMAF (Multimodal Fusion)
            # =================================================================
            logger.info(f"{Fore.YELLOW}Step 4: UMAF (Multimodal Fusion)")
            
            # Memory before Step 4
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            mem_step4_start = process.memory_info().rss / 1024 / 1024
            
            step4_start = time.time()
            
            # Fusion and classification
            fused_outputs = trainer.model.umaf(text_cml, vision_cml, audio_cml)
            logits = fused_outputs['logits']
            _, preds = torch.max(logits, 1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step4_time = time.time() - step4_start
            
            # Memory after Step 4
            mem_step4_end = process.memory_info().rss / 1024 / 1024
            if torch.cuda.is_available():
                gpu_mem_step4 = torch.cuda.max_memory_allocated() / 1024 / 1024
                step4_memory = (mem_step4_end - mem_step4_start) + gpu_mem_step4
            else:
                step4_memory = mem_step4_end - mem_step4_start
            
            step_times['Step 4: UMAF'] = step4_time
            step_memories['Step 4: UMAF'] = step4_memory
            step_latencies['Step 4: UMAF'] = step4_time / batch_size
            step_qps['Step 4: UMAF'] = batch_size / step4_time if step4_time > 0 else 0
            
            logger.info(f"{Fore.GREEN}  Time: {step4_time:.4f}s | Latency: {step_latencies['Step 4: UMAF']:.6f}s/sample | QPS: {step_qps['Step 4: UMAF']:.2f} | Memory: {step4_memory:.2f}MB\n")
        
        # Total time
        total_time = sum([
            step_times['Step 1: FAME'],
            step_times['Step 2: R³GC'],
            step_times['Step 3: CMEL/PC-CML'],
            step_times['Step 4: UMAF']
        ])
        
        # Memory measurement
        memory_end = process.memory_info().rss / 1024 / 1024
        memory_used = memory_end - memory_start
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            total_memory = memory_used + gpu_memory
        else:
            total_memory = memory_used
        
        # Per-sample metrics
        latency_per_sample = total_time / batch_size
        qps = batch_size / total_time if total_time > 0 else 0
        
        # Print summary
        logger.info(f"{Fore.MAGENTA}{'='*100}")
        logger.info(f"{Fore.MAGENTA}SUMMARY - {dataset_name}")
        logger.info(f"{Fore.MAGENTA}{'='*100}")
        logger.info(f"{Fore.GREEN}Total Time:          {total_time:.4f}s")
        logger.info(f"{Fore.GREEN}Latency/sample:      {latency_per_sample:.6f}s")
        logger.info(f"{Fore.GREEN}QPS:                 {qps:.2f} samples/sec")
        logger.info(f"{Fore.GREEN}Memory Used:         {total_memory:.2f} MB")
        logger.info(f"{Fore.GREEN}Batch Size:          {batch_size}")
        logger.info(f"{Fore.MAGENTA}{'='*100}\n")
        
        # Store results with per-step metrics
        result = {
            'Dataset': dataset_name,
            'Step 1 Time (s)': f'{step_times["Step 1: FAME"]:.4f}',
            'Step 1 Latency (s)': f'{step_latencies["Step 1: FAME"]:.6f}',
            'Step 1 QPS': f'{step_qps["Step 1: FAME"]:.2f}',
            'Step 1 Memory (MB)': f'{step_memories["Step 1: FAME"]:.2f}',
            'Step 2 Time (s)': f'{step_times["Step 2: R³GC"]:.4f}',
            'Step 2 Latency (s)': f'{step_latencies["Step 2: R³GC"]:.6f}',
            'Step 2 QPS': f'{step_qps["Step 2: R³GC"]:.2f}',
            'Step 2 Memory (MB)': f'{step_memories["Step 2: R³GC"]:.2f}',
            'Step 3 Time (s)': f'{step_times["Step 3: CMEL/PC-CML"]:.4f}',
            'Step 3 Latency (s)': f'{step_latencies["Step 3: CMEL/PC-CML"]:.6f}',
            'Step 3 QPS': f'{step_qps["Step 3: CMEL/PC-CML"]:.2f}',
            'Step 3 Memory (MB)': f'{step_memories["Step 3: CMEL/PC-CML"]:.2f}',
            'Step 4 Time (s)': f'{step_times["Step 4: UMAF"]:.4f}',
            'Step 4 Latency (s)': f'{step_latencies["Step 4: UMAF"]:.6f}',
            'Step 4 QPS': f'{step_qps["Step 4: UMAF"]:.2f}',
            'Step 4 Memory (MB)': f'{step_memories["Step 4: UMAF"]:.2f}',
            'Total Time (s)': f'{total_time:.4f}',
            'Total Latency (s)': f'{latency_per_sample:.6f}',
            'Total QPS': f'{qps:.2f}',
            'Total Memory (MB)': f'{total_memory:.2f}'
        }
        
        # Summary table (for easier viewing)
        summary_result = {
            'Dataset': dataset_name,
            'Step 1: FAME (s)': f'{step_times["Step 1: FAME"]:.4f}',
            'Step 2: R³GC (s)': f'{step_times["Step 2: R³GC"]:.4f}',
            'Step 3: CMEL/PC-CML (s)': f'{step_times["Step 3: CMEL/PC-CML"]:.4f}',
            'Step 4: UMAF (s)': f'{step_times["Step 4: UMAF"]:.4f}',
            'Total Time (s)': f'{total_time:.4f}',
            'Total Latency (s)': f'{latency_per_sample:.6f}',
            'Total QPS': f'{qps:.2f}',
            'Total Memory (MB)': f'{total_memory:.2f}'
        }
        
        return result, summary_result
    
    def run_all(self):
        """Measure all datasets"""
        base_dir = Path.cwd()
        script_dir = base_dir / 'src' if (base_dir / 'src').exists() else base_dir
        
        datasets = [
            {
                'name': 'HateMM',
                'script': str(script_dir / 'CL_based_recent_short.py'),
                'config': 'config'
            },
            {
                'name': 'MHClip-EN',
                'script': str(script_dir / 'CL_Multiclip_en.py'),
                'config': 'config'
            },
            {
                'name': 'MHClip-ZH',
                'script': str(script_dir / 'CL_Multiclip_zn.py'),
                'config': 'config_zn'
            }
        ]
        
        summary_results = []
        detailed_results = []
        
        for dataset in datasets:
            try:
                detailed, summary = self.measure_step_by_step(
                    dataset['name'], 
                    dataset['script'], 
                    dataset['config']
                )
                detailed_results.append(detailed)
                summary_results.append(summary)
            except Exception as e:
                logger.error(f"{Fore.RED}Error measuring {dataset['name']}: {e}")
                import traceback
                traceback.print_exc()
                logger.error(f"{Fore.RED}Skipping...\n")
        
        self.print_final_tables(detailed_results, summary_results)
    
    def print_final_tables(self, detailed_results, summary_results):
        """Print final comparison tables"""
        logger.info(f"\n{Fore.CYAN}{'='*150}")
        logger.info(f"{Fore.CYAN}FINAL RESULTS - DETAILED PER-STEP METRICS")
        logger.info(f"{Fore.CYAN}{'='*150}\n")
        
        # Table 1: Per-step detailed metrics
        logger.info(f"{Fore.MAGENTA}Table 1: Per-Step Performance (Time, Latency, QPS, Memory)")
        logger.info(f"{Fore.MAGENTA}{'-'*150}")
        df_detailed = pd.DataFrame(detailed_results)
        
        # Print in readable format
        for _, row in df_detailed.iterrows():
            logger.info(f"\n{Fore.CYAN}Dataset: {row['Dataset']}")
            logger.info(f"{Fore.WHITE}  Step 1 (FAME):       Time={row['Step 1 Time (s)']:<8} | Latency={row['Step 1 Latency (s)']:<10} | QPS={row['Step 1 QPS']:<8} | Memory={row['Step 1 Memory (MB)']}")
            logger.info(f"{Fore.WHITE}  Step 2 (R³GC):       Time={row['Step 2 Time (s)']:<8} | Latency={row['Step 2 Latency (s)']:<10} | QPS={row['Step 2 QPS']:<8} | Memory={row['Step 2 Memory (MB)']}")
            logger.info(f"{Fore.WHITE}  Step 3 (CMEL/PC-CML): Time={row['Step 3 Time (s)']:<8} | Latency={row['Step 3 Latency (s)']:<10} | QPS={row['Step 3 QPS']:<8} | Memory={row['Step 3 Memory (MB)']}")
            logger.info(f"{Fore.WHITE}  Step 4 (UMAF):       Time={row['Step 4 Time (s)']:<8} | Latency={row['Step 4 Latency (s)']:<10} | QPS={row['Step 4 QPS']:<8} | Memory={row['Step 4 Memory (MB)']}")
            logger.info(f"{Fore.GREEN}  TOTAL:               Time={row['Total Time (s)']:<8} | Latency={row['Total Latency (s)']:<10} | QPS={row['Total QPS']:<8} | Memory={row['Total Memory (MB)']}")
        
        # Table 2: Summary comparison
        logger.info(f"\n{Fore.MAGENTA}Table 2: Summary Comparison")
        logger.info(f"{Fore.MAGENTA}{'-'*150}")
        df_summary = pd.DataFrame(summary_results)
        logger.info(f"\n{df_summary.to_string(index=False)}\n")
        
        # Save to CSV
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        detailed_path = Path(f'performance_detailed_{timestamp}.csv')
        summary_path = Path(f'performance_summary_{timestamp}.csv')
        
        df_detailed.to_csv(detailed_path, index=False)
        df_summary.to_csv(summary_path, index=False)
        
        logger.info(f"{Fore.GREEN}✓ Detailed results saved to: {detailed_path}")
        logger.info(f"{Fore.GREEN}✓ Summary saved to: {summary_path}\n")
        
        # LaTeX table - Per-step time
        logger.info(f"{Fore.YELLOW}LaTeX Table - Time per Step:")
        logger.info(f"{Fore.YELLOW}\\begin{{tabular}}{{lcccccc}}")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}Dataset & FAME (s) & R³GC (s) & CMEL/PC-CML (s) & UMAF (s) & Total (s) \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        for _, row in df_summary.iterrows():
            logger.info(f"{Fore.YELLOW}{row['Dataset']} & {row['Step 1: FAME (s)']} & {row['Step 2: R³GC (s)']} & {row['Step 3: CMEL/PC-CML (s)']} & {row['Step 4: UMAF (s)']} & {row['Total Time (s)']} \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}\\end{{tabular}}\n")
        
        # LaTeX table - Per-step QPS
        logger.info(f"{Fore.YELLOW}LaTeX Table - QPS per Step:")
        logger.info(f"{Fore.YELLOW}\\begin{{tabular}}{{lcccccc}}")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}Dataset & FAME & R³GC & CMEL/PC-CML & UMAF & Total \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        for _, row in df_detailed.iterrows():
            logger.info(f"{Fore.YELLOW}{row['Dataset']} & {row['Step 1 QPS']} & {row['Step 2 QPS']} & {row['Step 3 QPS']} & {row['Step 4 QPS']} & {row['Total QPS']} \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}\\end{{tabular}}\n")
        
        # LaTeX table - Per-step Memory
        logger.info(f"{Fore.YELLOW}LaTeX Table - Memory per Step (MB):")
        logger.info(f"{Fore.YELLOW}\\begin{{tabular}}{{lcccccc}}")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}Dataset & FAME & R³GC & CMEL/PC-CML & UMAF & Total \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        for _, row in df_detailed.iterrows():
            logger.info(f"{Fore.YELLOW}{row['Dataset']} & {row['Step 1 Memory (MB)']} & {row['Step 2 Memory (MB)']} & {row['Step 3 Memory (MB)']} & {row['Step 4 Memory (MB)']} & {row['Total Memory (MB)']} \\\\")
        logger.info(f"{Fore.YELLOW}\\hline")
        logger.info(f"{Fore.YELLOW}\\end{{tabular}}\n")


def main():
    """Main entry"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    colorama.init()
    
    measurer = DetailedPerformanceMeasure()
    measurer.run_all()
    
    logger.info(f"{Fore.CYAN}{'='*100}")
    logger.info(f"{Fore.CYAN}DETAILED PERFORMANCE MEASUREMENT COMPLETED")
    logger.info(f"{Fore.CYAN}{'='*100}\n")


if __name__ == '__main__':
    main()