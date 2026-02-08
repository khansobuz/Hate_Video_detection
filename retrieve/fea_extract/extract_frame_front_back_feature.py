import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

dataset_dir = "data/MultiHateClip/zh"
frames_path = "frames_16_front"  # or "frames_16_front"
output_file = os.path.join(dataset_dir, f"fea/fea_{frames_path}_google-vit-base-16-224.pt")
model_id = "local_models/vit-base-patch16-224"


class MyDataset(Dataset):
    def __init__(self, dataset_dir, frames_path):
        self.dataset_dir = dataset_dir
        self.frames_path = frames_path
        vid_file = os.path.join(dataset_dir, "vids.csv")
        with open(vid_file, "r") as f:
            self.vids = [line.strip() for line in f]

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(16):  # Get 16 frames
            frame_path = os.path.join(self.dataset_dir, self.frames_path, vid, f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            else:
                frames.append(Image.new("RGB", (224, 224), color="black"))
        return vid, frames


def collate_fn(batch, processor):
    vids, all_frames = zip(*batch)
    all_frames = [frame for frames in all_frames for frame in frames]
    processed_frames = processor(images=all_frames, return_tensors="pt", padding=True)
    return vids, processed_frames


if __name__ == "__main__":
    # Load model and processor
    model = AutoModel.from_pretrained(model_id, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    # Create dataset and dataloader
    dataset = MyDataset(dataset_dir, frames_path)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=0  # Use 0 on Windows to avoid multiprocessing issues
    )

    features = {}

    # Extract features
    model.eval()
    with torch.no_grad():
        for vids, processed_frames in tqdm(dataloader):
            bs = len(vids)
            inputs = {k: v.to(model.device) for k, v in processed_frames.items()}
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.view(bs, 16, -1, outputs.last_hidden_state.size(-1))
            for i, vid in enumerate(vids):
                features[vid] = hidden_states[i][:, 0].detach().cpu()

    # Save features
    torch.save(features, output_file)
    print(f"Saved features to {output_file}")
