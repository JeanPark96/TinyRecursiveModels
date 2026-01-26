import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import json, random
from collections import defaultdict

def split_charadessta_train_val(
    train_json_path: str,
    out_train_path: str,
    out_val_path: str,
    val_ratio: float = 0.1,
    seed: int = 123,
):
    random.seed(seed)

    data = json.load(open(train_json_path, "r"))
    by_vid = defaultdict(list)
    for ex in data:
        by_vid[ex["video_id"]].append(ex)

    vids = list(by_vid.keys())
    random.shuffle(vids)

    n_val = max(1, int(len(vids) * val_ratio))
    val_vids = set(vids[:n_val])
    train_vids = set(vids[n_val:])

    train_split = []
    val_split = []
    for vid, items in by_vid.items():
        (val_split if vid in val_vids else train_split).extend(items)

    json.dump(train_split, open(out_train_path, "w"), indent=2)
    json.dump(val_split, open(out_val_path, "w"), indent=2)

    print(f"unique videos: {len(vids)}")
    print(f"train videos: {len(train_vids)}, train samples: {len(train_split)}")
    print(f"val videos:   {len(val_vids)},   val samples:   {len(val_split)}")

@torch.no_grad()
def precompute_charades_text_emb(
    ann_json_path: str,
    out_npy_path: str,
    out_tokens_path: str, # NEW: Path for sequence tokens
    out_mask_path: str,   # NEW: Path for attention mask
    model_name="openai/clip-vit-base-patch32",
    batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fp16=True,
):
    with open(ann_json_path, "r") as f:
        items = json.load(f)

    texts = [ex["query"] for ex in items]

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    # Configs
    D = int(model.config.projection_dim)
    max_len = model.config.text_config.max_position_embeddings # Usually 77
    # Define dtypes correctly for both libraries
    dtype_np = np.float16 if fp16 else np.float32
    dtype_torch = torch.float16 if fp16 else torch.float32
    
    # Pre-allocate arrays
    # Pooled: (N, D)
    out_pooled = np.empty((len(texts), D), dtype=dtype_np)
    # Tokens: (N, L, D) - Projected sequence features
    out_tokens = np.empty((len(texts), max_len, D), dtype=dtype_np)
    # Mask: (N, L) - Bool or Int
    out_mask = np.empty((len(texts), max_len), dtype=bool)

    use_amp = (device == "cuda") and fp16

    for s in tqdm(range(0, len(texts), batch_size), desc="Text batches"):
        e = min(len(texts), s + batch_size)
        batch = texts[s:e]
        
        # Tokenize with padding to max_length (77) to keep consistent shape
        inputs = proc(
            text=batch, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=max_len, 
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # 1. Get Text Encoder Outputs
            # CLIPModel's text_model returns last_hidden_state (batch, seq_len, hidden)
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # 2. Process Sequence Tokens
            # The text_model output is usually the hidden state *before* projection.
            # We project it to align with the visual space (512 dim).
            # shape: (B, L, H) -> (B, L, D)
            seq_hidden = text_outputs.last_hidden_state
            seq_projected = model.text_projection(seq_hidden)
            
            # 3. Process Pooled Embedding (Original logic)
            # We can grab it from 'get_text_features' or compute manually. 
            # Re-running get_text_features is safer to match CLIP's exact pooling/norm logic.
            pooled_feats = model.get_text_features(**inputs)
            pooled_feats = pooled_feats / pooled_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Save to numpy
        # out_pooled[s:e] = pooled_feats.detach().cpu().to(dtype_np if fp16 else torch.float32).numpy()
        # out_tokens[s:e] = seq_projected.detach().cpu().to(dtype_np if fp16 else torch.float32).numpy()
        # out_mask[s:e] = inputs["attention_mask"].detach().cpu().bool().numpy()
        # FIX: Use dtype_torch for .to()
        out_pooled[s:e] = pooled_feats.detach().cpu().to(dtype_torch).numpy()
        out_tokens[s:e] = seq_projected.detach().cpu().to(dtype_torch).numpy()
        out_mask[s:e] = inputs["attention_mask"].detach().cpu().bool().numpy()
    # Save files
    np.save(out_npy_path, out_pooled)
    np.save(out_tokens_path, out_tokens)
    np.save(out_mask_path, out_mask)
    
    print(f"Saved pooled: {out_npy_path} {out_pooled.shape}")
    print(f"Saved tokens: {out_tokens_path} {out_tokens.shape}")
    print(f"Saved mask:   {out_mask_path}   {out_mask.shape}")

import numpy as np
import torch

def custom_collate(batch, Tmax = 256):
    """
    batch: list of dicts from dataset:
      {
        "video_id": str,
        "query": str,
        "start_sec": float,
        "end_sec": float,
        "i0": int,
        "i1": int,
        "video_emb": (T, D) np array (maybe memmap slice),
        "text_emb": (D,) np array
        "query_tokens" : (B, L, D),
        "query_mask" : (B, L)
      }

    returns dict with:
      video_emb: (B, Tmax, D) float32 torch
      video_mask: (B, Tmax) bool torch
      text_emb: (B, D) float32 torch
      query_tokens: (B, L, D)     <- NEW: Sequence
      query_mask: (B, L)          <- NEW: Sequence mask
      plus metadata lists
    """
    B = len(batch)
    lengths = [int(x["video_emb"].shape[0]) for x in batch]
    #Tmax = max(lengths) if lengths else 0
    D = int(batch[0]["video_emb"].shape[1]) if Tmax > 0 else int(batch[0]["text_emb"].shape[0])

    # Allocate (float32 is typical for training)
    video = torch.zeros((B, Tmax, D), dtype=torch.float32)
    mask = torch.zeros((B, Tmax), dtype=torch.bool)

    # Text embeddings
    text = torch.stack([
        torch.as_tensor(x["text_emb"], dtype=torch.float32)
        for x in batch
    ])  # (B, D)

    # Stack sequence tokens
    query_tokens = torch.stack([torch.as_tensor(x["query_tokens"], dtype=torch.float32) for x in batch])
    
    # Stack sequence masks
    query_mask = torch.stack([torch.as_tensor(x["query_mask"], dtype=torch.bool) for x in batch])

    for i, x in enumerate(batch):
        v = x["video_emb"]
        t = int(v.shape[0])
        if t > 0:
            video[i, :t] = torch.as_tensor(v, dtype=torch.float32)
            mask[i, :t] = True
    # Stack numerical metadata into Tensors
    i0 = torch.tensor([x["i0"] for x in batch], dtype=torch.long)
    i1 = torch.tensor([x["i1"] for x in batch], dtype=torch.long)
    start_sec = torch.tensor([x["start_sec"] for x in batch], dtype=torch.float32)
    end_sec = torch.tensor([x["end_sec"] for x in batch], dtype=torch.float32)
    return {
        "video_emb": video,
        "video_mask": mask,
        "text_emb": text,
        "query_tokens": query_tokens, # (B, 77, D)
        "query_mask": query_mask,     # (B, 77)
        "lengths": torch.tensor(lengths, dtype=torch.long),

        # metadata (keep as python lists)
        "video_id": [x["video_id"] for x in batch],
        "query": [x["query"] for x in batch],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "i0": i0,
        "i1": i1,
    }


class CharadesSTA(Dataset):
    def __init__(
        self,
        ann_json: str,
        volume_embed_root: str,
        text_emb_path: str,
        text_tokens_path: str,              # <-- NEW
        text_mask_path: str = None,         # <-- NEW (Optional, usually side-by-side with tokens)
        cache_root: str = "/local_disk0/embed_cache",
        mmap_video: bool = True,
        mmap_text: bool = False,
        return_numpy: bool = True,
        min_seg_len: int = 1,
    ):
        self.ann_path = Path(ann_json)
        self.volume_embed_root = Path(volume_embed_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.mmap_video = mmap_video
        self.mmap_text = mmap_text
        self.return_numpy = return_numpy
        self.min_seg_len = int(min_seg_len)

        # ---- load annotations ----
        with open(self.ann_path, "r") as f:
            data = json.load(f)
        self.items: List[Dict[str, Any]] = []
        for ex in data:
            self.items.append({
                "video_id": str(ex["video_id"]),
                "query": str(ex["query"]),
                "start": float(ex["start"]),
                "end": float(ex["end"]),
            })

        # ---- load precomputed POOLED text embeddings ----
        self.text_emb_path = Path(text_emb_path)
        if not self.text_emb_path.exists():
            raise FileNotFoundError(f"text_emb_path not found: {self.text_emb_path}")
        
        self.text_emb = np.load(str(self.text_emb_path), mmap_mode="r" if self.mmap_text else None)

        # ---- load precomputed TOKEN sequence embeddings ----
        self.text_tokens_path = Path(text_tokens_path)
        if not self.text_tokens_path.exists():
             raise FileNotFoundError(f"text_tokens_path not found: {self.text_tokens_path}")
        
        # Determine mask path if not provided
        if text_mask_path is None:
            # Assume it's named like '..._tokens.npy' -> '..._mask.npy'
            # Or just pass it explicitly in load_dataset
            text_mask_path = str(self.text_tokens_path).replace("tokens.npy", "mask.npy")
            if text_mask_path == str(self.text_tokens_path):
                 text_mask_path = str(self.text_tokens_path).replace(".npy", "_mask.npy")

        self.text_mask_path = Path(text_mask_path)
        if not self.text_mask_path.exists():
             raise FileNotFoundError(f"text_mask_path not found: {self.text_mask_path}")

        # Load tokens and mask
        self.text_tokens = np.load(str(self.text_tokens_path), mmap_mode="r" if self.mmap_text else None)
        self.text_mask = np.load(str(self.text_mask_path), mmap_mode="r" if self.mmap_text else None)

        # Consistency Check
        if not (len(self.text_emb) == len(self.text_tokens) == len(self.items)):
            raise ValueError("Row mismatch between annotations, pooled embeddings, and tokens.")

    def _ensure_local_video_dir(self, video_id: str) -> Path:
        src = self.volume_embed_root / video_id
        if not src.exists():
             # Fallback logic or raise error
             pass 
        dst = self.cache_root / video_id
        if dst.exists(): return dst
        tmp = self.cache_root / f"{video_id}.tmp"
        if tmp.exists(): shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        os.replace(tmp, dst)
        return dst

    def _time_to_index_range(self, local_video_dir: Path, start_s: float, end_s: float) -> Tuple[int, int]:
        # (Same as your existing logic)
        ts_path = local_video_dir / "timestamps_sec.npy"
        # ... [Your existing implementation] ...
        # (Assuming you keep the implementation from your provided snippet)
        # For brevity, reusing the simple logic:
        if ts_path.exists():
            ts = np.load(str(ts_path))
            i0 = int(np.searchsorted(ts, start_s, side="left"))
            i1 = int(np.searchsorted(ts, end_s, side="right"))
        else:
            i0, i1 = 0, 10 # dummy fallback if logic not copied
        
        # Clamp
        emb_path = local_video_dir / "clip_embeddings_fp16.npy"
        if emb_path.exists():
            emb = np.load(str(emb_path), mmap_mode="r")
            T = len(emb)
            i0 = max(0, min(i0, T))
            i1 = max(0, min(i1, T))
        return i0, i1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.items[i]
        local_dir = self._ensure_local_video_dir(ex["video_id"])
        
        # Video
        emb_path = local_dir / "clip_embeddings_fp16.npy"
        i0, i1 = self._time_to_index_range(local_dir, ex["start"], ex["end"])
        
        if emb_path.exists():
            emb = np.load(str(emb_path), mmap_mode="r" if self.mmap_video else None)
            seg = emb[i0:i1]
        else:
            seg = np.zeros((1, 512), dtype=np.float32) # Fallback

        if not self.return_numpy:
            seg = np.array(seg, dtype=np.float32)

        # Text
        text_emb = self.text_emb[i]         # Pooled (D,)
        text_tokens = self.text_tokens[i]   # Sequence (L, D)
        text_mask = self.text_mask[i]       # Mask (L,)

        return {
            "video_id": ex["video_id"],
            "query": ex["query"],
            "start_sec": ex["start"],
            "end_sec": ex["end"],
            "i0": i0,
            "i1": i1,
            "video_emb": seg,
            "text_emb": text_emb,
            "query_tokens": text_tokens, # <-- New
            "query_mask": text_mask      # <-- New
        }
def worker_init_fn(worker_id, seed=123):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
def load_dataset(batch_size=16):
    print("Loading Dataset...")
    
    #raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'
    # Common root
    feat_root = "/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/Extracted/clip-vit-base-patch32-json/"
    
    # Helper to define paths compactly
    def get_paths(split_name):
        return {
            "text_emb_path":    f"{feat_root}/{split_name}.npy",
            "text_tokens_path": f"{feat_root}/{split_name}_tokens.npy", # Assumes you generated this
            "text_mask_path":   f"{feat_root}/{split_name}_mask.npy"     # Assumes you generated this
        }

    train_ds = CharadesSTA(
        ann_json="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/charades_sta_train_split.json",
        volume_embed_root="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/Extracted/clip-vit-base-patch32/",
        **get_paths("train_split"),
        cache_root="/local_disk0/embed_cache",
    )

    val_ds = CharadesSTA(
        ann_json="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/charades_sta_val_split.json",
        volume_embed_root="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/Extracted/clip-vit-base-patch32/",
        **get_paths("val_split"),
        cache_root="/local_disk0/embed_cache",
    )

    test_ds = CharadesSTA(
        ann_json="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/charades_sta_test.json",
        volume_embed_root="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/open_source_video_datasets/Charades-STA/Extracted/clip-vit-base-patch32/",
        **get_paths("test"),
        cache_root="/local_disk0/embed_cache",
    )

    tr_dataloader = DataLoader(train_ds, 
                               batch_size=batch_size, 
                               shuffle=True, 
                               collate_fn=custom_collate, 
                               drop_last=True,
                                pin_memory=True,
                                persistent_workers=False,)
    val_dataloader = DataLoader(val_ds, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=custom_collate,
                                pin_memory=True,
                                persistent_workers=False,)
    test_dataloader = DataLoader(test_ds, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 collate_fn=custom_collate,
                                pin_memory=True,
                                persistent_workers=False,)

    return train_ds, val_ds, test_ds, tr_dataloader, val_dataloader, test_dataloader
