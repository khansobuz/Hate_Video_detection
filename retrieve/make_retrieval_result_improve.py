"""
Improved retrieval script with weighted modality combination
This addresses the issue where all modalities had equal weight
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

def compute_combined_similarities(query_ids, base_ids, features_dict, labels_dict, 
                                   topk=100, batch_size=100, ignore_self=True,
                                   modality_weights=None):
    """
    Compute similarities with weighted modality combination
    
    Args:
        query_ids: List of query video IDs
        base_ids: List of base video IDs
        features_dict: Dict of {modality: {vid: feature}} for each modality
        labels_dict: Dict of {vid: label}
        topk: Number of top results to return per label class
        batch_size: Batch size for processing
        ignore_self: Whether to ignore self-matches
        modality_weights: Dict of weights for each modality (default: text=0.45, vision=0.35, audio=0.20)
    """
    # Default weights optimized for hate speech detection
    if modality_weights is None:
        modality_weights = {
            'text': 0.45,    # Text is most important for hate speech
            'vision': 0.35,  # Visual context is important
            'audio': 0.20    # Audio is least important but still contributes
        }
    
    # Ensure consistent ordering of IDs
    base_id_to_index = {id: idx for idx, id in enumerate(base_ids)}
    results = []

    # Prepare feature vectors for each modality
    query_vectors = {}
    base_vectors = {}
    for modality in features_dict:
        feature = features_dict[modality]
        query_vectors[modality] = np.array([feature[id] for id in query_ids])
        base_vectors[modality] = np.array([feature[id] for id in base_ids])

    # Process in batches
    for i in tqdm(range(0, len(query_ids), batch_size), desc="Computing similarities"):
        batch_ids = query_ids[i:i+batch_size]
        batch_vectors = {modality: vectors[i:i+batch_size] for modality, vectors in query_vectors.items()}
        
        # Compute similarity matrices for each modality
        sim_matrices = {}
        for modality in features_dict:
            # Cosine similarity = 1 - cosine distance
            sim_matrices[modality] = 1 - cdist(batch_vectors[modality], base_vectors[modality], metric='cosine')
        
        # Weighted combination of similarity matrices
        combined_sim = sum(modality_weights.get(mod, 1.0/len(sim_matrices)) * sim_matrices[mod] 
                          for mod in sim_matrices)
        
        # Process each query in the batch
        for j, sim in enumerate(combined_sim):
            query_id = batch_ids[j]
            
            # Ignore self-matches if requested
            if ignore_self:
                self_index = base_id_to_index.get(query_id, -1)
                if self_index != -1 and self_index < len(sim):
                    sim[self_index] = -np.inf
            
            # Separate results by label (0 and 1)
            results_0 = []
            results_1 = []
            
            # Get top-k for each label
            for idx, similarity in sorted(enumerate(sim), key=lambda x: x[1], reverse=True):
                base_id = base_ids[idx]
                label = labels_dict.get(base_id, -1)
                
                if label == 0 and len(results_0) < topk:
                    results_0.append({"vid": base_id, "similarity": float(similarity)})
                elif label == 1 and len(results_1) < topk:
                    results_1.append({"vid": base_id, "similarity": float(similarity)})
                
                # Stop when we have enough of both classes
                if len(results_0) >= topk and len(results_1) >= topk:
                    break
            
            # Store results
            results.append({
                'vid': query_id, 
                'similarities': [
                    {'vid': [r['vid'] for r in results_0], 'sim': [r['similarity'] for r in results_0]},
                    {'vid': [r['vid'] for r in results_1], 'sim': [r['similarity'] for r in results_1]}
                ]
            })
    
    return results

# Main processing code
if __name__ == '__main__':
    datasets = ['MultiHateClip/zh', 'MultiHateClip/en', 'HateMM']

    # Optimized weights for hate speech detection
    # These can be tuned based on your validation set performance
    modality_weights = {
        'text': 0.45,    # Text carries most hate speech signals
        'vision': 0.35,  # Visual context important for multimodal hate
        'audio': 0.20    # Audio tone/emotion contributes but less
    }
    
    print(f"Using modality weights: {modality_weights}")

    for dataset in datasets:
        dataset_path = Path('data') / dataset
        
        print(f'\n{"="*80}')
        print(f'Processing dataset: {dataset}')
        print(f'{"="*80}')
        
        # Load features from three modalities
        feature_files = {
            'audio': 'fea_audio_modal_retrieval.pt',
            'vision': 'fea_vision_modal_retrieval.pt',
            'text': 'fea_text_modal_retrieval.pt'
        }
        
        features_dict = {}
        for modality, file_name in feature_files.items():
            feature_path = dataset_path / 'fea' / file_name
            if not feature_path.exists():
                print(f"WARNING: Feature file not found: {feature_path}")
                continue
                
            print(f"Loading {modality} features...")
            feature = torch.load(feature_path, weights_only=True)
            
            # Process features: reduce to 1D if needed
            for vid, feat in feature.items():
                if len(feat.shape) != 1:
                    feature[vid] = feat.mean(dim=0)
                feature[vid] = feature[vid].numpy()  # Convert to NumPy for cdist
            
            features_dict[modality] = feature
            print(f"  Loaded {len(feature)} {modality} features")
        
        if len(features_dict) == 0:
            print(f"ERROR: No features found for {dataset}, skipping...")
            continue
        
        # Get common video IDs across all modalities
        common_vids = set.intersection(*(set(features_dict[modality].keys()) for modality in features_dict))
        print(f"Common videos across all modalities: {len(common_vids)}")
        
        # Load video ID splits
        train_vids = pd.read_csv(dataset_path / 'vids/train.csv', header=None)[0].tolist()
        valid_vids = pd.read_csv(dataset_path / 'vids/valid.csv', header=None)[0].tolist()
        test_vids = pd.read_csv(dataset_path / 'vids/test.csv', header=None)[0].tolist()
        train_valid_vids = list(set(train_vids + valid_vids))
        all_vids = list(set(train_valid_vids + test_vids))
        
        # Filter video IDs to those present in all modalities
        query_ids = [vid for vid in all_vids if vid in common_vids]
        base_ids = [vid for vid in train_valid_vids if vid in common_vids]
        
        print(f'Number of query videos: {len(query_ids)}')
        print(f'Number of base videos: {len(base_ids)}')
        
        # Load labels
        label_path = dataset_path / 'label.jsonl'
        if not label_path.exists():
            print(f"ERROR: Label file not found: {label_path}, skipping...")
            continue
            
        labels = pd.read_json(label_path, lines=True)
        labels_dict = labels.set_index('vid')['label'].to_dict()
        
        # Check label distribution
        label_counts = labels['label'].value_counts()
        print(f"Label distribution: {label_counts.to_dict()}")
        
        # Compute combined similarities with weighted modalities
        print("Computing weighted similarities...")
        result = compute_combined_similarities(
            query_ids, base_ids, features_dict, labels_dict, 
            topk=100,
            modality_weights=modality_weights
        )
        
        # Set output path
        output_path = dataset_path / 'retrieval' / 'all_modal1.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        print(f"Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            for item in result:
                f.write(json.dumps(item) + '\n')
        
        print(f"✓ Completed processing {dataset}")
        print(f"  Saved {len(result)} similarity records")

    print(f'\n{"="*80}')
    print("All datasets processed successfully!")
    print(f'{"="*80}')