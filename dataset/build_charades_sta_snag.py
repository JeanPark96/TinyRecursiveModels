from collections import OrderedDict
from copy import deepcopy
from functools import partial
import json
import math
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Custom GloVeTokenizer if torchtext does not work
class GloVeTokenizer:
    def __init__(self, name='6B', dim=300, cache_dir='/home/hlpark/snag_release/.vector_cache'):
        self.dim = dim
        self.cache_dir = cache_dir
        self.glove_map = {}
        
        # 1. Ensure Cache Directory Exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 2. Check for File (e.g., glove.6B.300d.txt)
        filename = f'glove.{name}.{dim}d.txt'
        file_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[GloVe] Downloading {name} vectors (this may take a moment)...")
            zip_name = f'glove.{name}.zip'
            zip_path = os.path.join(cache_dir, zip_name)
            
            # URL for standard GloVe vectors
            url = f'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.{name}.zip'
            
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
                print("[GloVe] Download and extraction complete.")
            except Exception as e:
                print(f"[GloVe] Error downloading: {e}. Please copy {filename} to {cache_dir} manually.")
        
        # 3. Load Vectors into Memory (Standard GloVe Loading)
        print(f"[GloVe] Loading vectors from {file_path}...")
        self.unk_vec = torch.zeros(dim) 
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    # vectors are usually float32
                    vec = torch.tensor([float(x) for x in parts[1:]])
                    self.glove_map[word] = vec
            print(f"[GloVe] Loaded {len(self.glove_map)} words.")
        except FileNotFoundError:
            print(f"[GloVe] Warning: File not found. Tokenizer will return zeros.")

    def __call__(self, text, max_len=None):
        # Basic Tokenization used by SnAG (split by space, lowercase)
        words = text.lower().replace('.', ' .').replace(',', ' ,').split()
        
        vecs = []
        for w in words:
            # Look up word, return 0-vector if unknown
            vecs.append(self.glove_map.get(w, self.unk_vec))
            
        if len(vecs) == 0:
            vecs = [self.unk_vec]

        # Stack to (Time, Channel)
        feats = torch.stack(vecs) 

        # Truncate if needed
        if max_len is not None and feats.shape[0] > max_len:
            feats = feats[:max_len]
            
        # Transpose to (Channel, Time) to match SnAG format
        feats = feats.transpose(0, 1)
        
        return feats

#SnAG provided Tokenizer
# import torchtext
# from torchtext.data import get_tokenizer
# from torchtext.vocab import GloVe
# class GloVeTokenizer:

#     def __init__(self, name='6B'):

#         self.vocab = GloVe(name=name)
#         self.tokenizer = get_tokenizer("basic_english")

#     def __call__(self, text, max_len=None):
#         """
#         Args:
#             text (str): text query.
#             max_len (int): maximum sequence length.

#         Returns:
#             feats (float tensor, (c, t)): feature sequence.
#         """
#         # tokenize by word
#         ## NOTE: unknown words are assigned zero vector
#         words = self.tokenizer(text)
#         feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
#         if max_len is not None:
#             feats = feats[:max_len]
#         feats = feats.transpose(0, 1)   # (c, t)

#         return feats

class BaseDataset(Dataset):

    def __init__(
        self,
        split,                  # data split, a tuple/list allowing concat of subsets
        is_training,            # whether in training mode
        
        anno_file,              # annotation json file
        vid_feat_dir,           # video feature directory
        text_feat_dir,          # text feature directory
        ext_score_dir,          # external score directory
        tokenizer,              # tokenizer (optional)

        max_vid_len,            # max video length (#clips) in training
        max_text_len,           # max text length (#tokens) in training
        clip_size,              # number of frames per clip / feature
        clip_stride,            # temporal stride of clips (in frame)
        downsample_rate=1,      # down-sampling rate for video features
        to_fixed_len=False,     # whether to resize video features to max length
        
        normalize_vid=False,    # whether to normalize video features to unit length
        normalize_text=False,   # whether to normalize text features to unit length
        normalize_scores=True,  # whether to normalize external score using sigmoid
        temperature=1.0,        # sigmoid temperature for score normalization

        crop_ratio=(0.9, 1.0),  # random cropping of video features in training
        trunc_thresh=0.5,       # threshold for event truncation in training
        max_num_text=None,      # max number of text queries per video in training
        
        group_method="greedy",  # text grouping method ("greedy" | "random" | "all")
        num_epochs=1,           # number of epochs
    ):
        super(BaseDataset, self).__init__()

        assert os.path.exists(anno_file)
        if not isinstance(split, (list, tuple)):
            split = (split, )
        if not isinstance(vid_feat_dir, (list, tuple)):
            vid_feat_dir = (vid_feat_dir, )
        assert all([os.path.isdir(d) for d in vid_feat_dir])
        if tokenizer is None:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0  # this must be updated upon starting a new epoch

        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        self.ext_score_dir = ext_score_dir
        self.tokenizer = tokenizer

        self.max_vid_len = max_vid_len
        self.max_text_len = max_text_len
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate
        self.to_fixed_len = to_fixed_len

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.normalize_scores = normalize_scores
        self.temperature = temperature

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh
        self.max_num_text = max_num_text

        self.vid_dict, self.text_dict = self._parse_annotations()
        
        self.group_method = group_method
        self.num_epochs = num_epochs

    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        # combine data from all splits
        anno_db = dict()
        for s in self.split:
            #assert s in anno, 'split [{:s}] does not exist'.format(s)
            anno_db.update(anno[s])

        vid_dict, text_dict = OrderedDict(), OrderedDict()
        for key, value in anno_db.items():
            if 'annotations' not in value:
                continue

            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps
            
            if 'num_clips' in value:
                num_clips = (
                    value['num_clips'] + self.downsample_rate - 1
                ) // self.downsample_rate
            else:
                num_clips = None

            text_ids, segments = tuple(), tuple()
            for s, pair in enumerate(value['annotations']):
                start = max(float(pair['segment'][0]), 0)
                end = min(float(pair['segment'][1]), duration)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                segment = (start, end)

                text = pair['sentence'].strip()
                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))
                text_ids += (text_id, )
                segments += (segment, )

                text_dict[text_id] = {
                    'text'      : text,
                    'segment'   : np.array(segment)[None],
                    'text_idx'  : s,
                    'vid_id'    : key,
                }
            
            if len(text_ids) == 0:
                continue

            vid_dict[key] = {
                'fps'       : fps,
                'num_frames': num_frames,
                'num_clips' : num_clips,
                'duration'  : duration,
                'text_ids'  : text_ids,
                'segments'  : np.array(segments),
            }

        return vid_dict, text_dict

    def _load_vid_feats(self, vid_id):
        try:
            vid_feat_files = [os.path.join(d, vid_id + '.npy') \
                for d in self.vid_feat_dir]
            vid_feats = [np.load(f).astype(np.float32) for f in vid_feat_files]
        except:
            raise ValueError(
                'failed to load features for video {:s}'.format(vid_id)
            )

        # assume features from different sources are apporoximately aligned
        # (flow features may be one unit shorter than RGB features)
        if len(vid_feats) > 1:
            feat_lens = [len(x) for x in vid_feats]
            max_len, min_len = max(feat_lens), min(feat_lens)
            assert max_len - min_len <= 1, \
                'misaligned features ([max] {:d}, [min] {:d}) for video {:s}' \
                ''.format(max_len, min_len, vid_id)

            # pad shorter sequences by replicating last feature vector
            for idx in range(len(vid_feats)):
                if feat_lens[idx] < max_len:
                    pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                    vid_feats[idx] = np.concatenate((vid_feats[idx], pad))

            # concatenate features along channel dimension
            vid_feats = np.concatenate(vid_feats, axis=-1)  # (t, c)
        else:
            vid_feats = vid_feats[0]

        # temporally down-sample features
        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)
        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,          # float tensor (c, t), full video features 
        segments,       # float tensor (n, 2), event segments
        offset,         # float, clip offset
        num_trials=5000 # int, number of trials
    ):
        vid_len = feats.size(1)
        max_vid_len = self.max_vid_len

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, segments

            max_vid_len = random.randint(
                max(np.ceil(self.crop_ratio[0] * vid_len), 1),
                min(np.ceil(self.crop_ratio[1] * vid_len), vid_len)
            )
            if max_vid_len == vid_len:
                return feats, segments

        # rough estimate on the range of valid chunks
        s0 = max(0, np.floor(segments[:, 0].max() - max_vid_len))
        s1 = min(vid_len - max_vid_len, np.ceil(segments[:, 1].min()))
        
        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        for _ in range(num_trials):
            ws = random.randint(s0, s1) # window start
            we = ws + max_vid_len       # window end

            # check overlap with segments
            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > self.trunc_thresh):
                feats = feats[:, ws:we]
                segments = torch.clamp(
                    segments - ws, min=-offset, max=we - ws + offset
                )
                return feats, segments

        raise ValueError('no valid truncation found')

    def _load_text_feats(self, text_id):
        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        else:
            try:
                text_feat_file = os.path.join(self.text_feat_dir, text_id + '.npy')
                text_feats = np.load(text_feat_file).astype(np.float32)
            except:
                raise ValueError(
                    'failed to load features for sentence {:s}'.format(text_id)
                )
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def _load_ext_scores(self, text_id):
        try:
            score_file = os.path.join(self.ext_score_dir, text_id + '.npy')
            scores = np.load(score_file).astype(np.float32)
        except:
            raise ValueError(
                'failed to load external scores for sentence {:s}'.format(text_id)
            )

        # temporally down-sample scores
        if self.downsample_rate > 1:
            scores = scores[::self.downsample_rate]

        scores = torch.from_numpy(np.ascontiguousarray(scores))[None]   # (1, t)

        if self.normalize_scores:
            scores = torch.sigmoid(scores / self.temperature)

        return scores

    def _avgpool_to_fixed_len(self, feats, size):
        vid_len = feats.size(1)
        sampling_ratio = math.ceil(vid_len / size)
        feats = F.interpolate(
            feats[None],
            size=size * sampling_ratio, mode='linear', align_corners=False
        )
        if sampling_ratio > 1:
            feats = F.avg_pool1d(feats, kernel_size=sampling_ratio)
        feats = feats[0]

        return feats

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()



class TextCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video-text pair (where the text serves as the probe) and optionally 
    includes addition text queries from the same video. The dataset size 
    is equal to the total number of text queries from all videos.

    Expected behavior:
    - train: a video + a single text query
    - eval: a video + a single text query
    """
    def __init__(self, **kwargs):
        super(TextCentricDataset, self).__init__(**kwargs)

        self.data_list = tuple(self.text_dict.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text_id = self.data_list[idx]
        text_dict = self.text_dict[text_id]
        vid_id = text_dict['vid_id']
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        vid_feats = self._load_vid_feats(vid_id)
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            text_dict['segment'] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        ## NOTE: use current text as probe
        if self.is_training and not self.to_fixed_len:
            vid_feats, segments = self._truncate_vid_feats(
                vid_feats, segments, clip_offset
            )

        # load text features
        text_feats = self._load_text_feats(text_id)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores = self._load_ext_scores(text_id)
            if self.to_fixed_len:
                ext_scores = self._avgpool_to_fixed_len(
                    ext_scores, self.max_vid_len
                )
            ext_scores = ext_scores[0]
        else:
            ext_scores = None
        
        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : text_dict['segment'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segment in grid unit
                 'sentence' : text_dict["text"],
                 'vid'        : vid_feats,              # video features (c2, t2)
                 'text'       : text_feats,             # text features (c1, t1)
                 'ext_scores' : ext_scores,             # external scores (t2, )
                }


# =========================================================
# 1. The Adapter Class (Inherits SOTA Logic)
# =========================================================
class CharadesSnagAdapter(TextCentricDataset):
    """
    Adapts the original SnAG TextCentricDataset to work with a custom
    GloVe tokenizer and dynamic arguments, ensuring fair comparison
    by reusing the original resizing/windowing logic.
    """
    def __init__(self, is_training=True, tokenizer=None, cache_ram=False, **kwargs):
        # 1. Capture the custom tokenizer
        self.custom_tokenizer = tokenizer
        self.cache_ram = cache_ram
        self.video_cache = {}
        defaults = {
            'is_training': is_training,       # Default to training mode
            'text_feat_dir': None,     # We use custom tokenizer, so no dir needed
            'ext_score_dir': None,     # Not using external scores
            'max_text_len': 32,        # Default max text length
            'tokenizer': tokenizer          # Pass None to parent, we handle tokenization ourselves
        }
        
        for key, default_val in defaults.items():
            if key not in kwargs:
                kwargs[key] = default_val

        # 3. Clean up args that might crash BaseDataset (like 'name')
        kwargs.pop('name', None)
        
        # 3. Call Original Init (Loads JSON, sets up video paths, etc.)
        super().__init__(**kwargs)
        # 3. RAM CACHING LOGIC (The Speedup)
        if self.cache_ram:
            print(f"[Dataset] Pre-loading video features into RAM ({self.split})...")
            # Iterate unique videos to avoid duplicate loading
            unique_vids = list(self.vid_dict.keys())
            
            for vid_id in tqdm(unique_vids, desc="Caching Videos"):
                try:
                    # A. Load Raw (Slow Disk I/O)
                    feat = self._load_vid_feats(vid_id)
                    orig_len = feat.size(1) # Save original length for stride calc
                    
                    # B. Resize Immediately (Save RAM & CPU later)
                    if self.to_fixed_len:
                        feat = self._avgpool_to_fixed_len(feat, self.max_vid_len)
                    
                    # C. Store tuple: (Feature, Original_Length)
                    self.video_cache[vid_id] = (feat, orig_len)
                    
                except Exception as e:
                    # Use a dummy if a file is corrupt, to prevent crash
                    print(f"Warning: Failed to cache {vid_id}: {e}")

            # Memory Check
            sample_tensor = next(iter(self.video_cache.values()))[0]
            mem_size_mb = (len(self.video_cache) * sample_tensor.element_size() * sample_tensor.numel()) / 1e6
            print(f"[Dataset] Cache Complete. Used approx {mem_size_mb:.2f} MB RAM.")

    def _load_text_feats(self, text_id):
        """
        OVERRIDE: Use custom tokenizer instead of loading .pkl files.
        Input: text_id (str)
        Output: Tensor (Channel, Tokens) to match SnAG convention
        """
        # Get raw sentence from the metadata loaded by parent
        #print(self.text_dict[text_id])
        raw_sentence = self.text_dict[text_id]['text']
        
        if self.custom_tokenizer is not None:
            # 2. Tokenize -> Returns shape (Channel=300, Length=N)
            feat = self.custom_tokenizer(raw_sentence) 
            
            # 3. FIX: Enforce Truncation
            # We strictly cut off any tokens beyond max_text_len
            if self.max_text_len is not None and feat.shape[1] > self.max_text_len:
                feat = feat[:, :self.max_text_len]
            return feat
        else:
            # Fallback to parent logic (loading from disk)
            return super()._load_text_feats(text_id)

    def __getitem__(self, idx):
        # 1. Get standard output from parent (performs resizing logic)
        data = super().__getitem__(idx)
        
        # 2. Inject extra metadata (useful for debugging/logging)
        text_id = self.data_list[idx]
        vid_id = self.text_dict[text_id]['vid_id']
        
        data['video_id'] = vid_id
        data['text_id'] = text_id
        
        return data
        # Fetch metadata
        # text_id = self.data_list[idx]
        # text_dict = self.text_dict[text_id]
        # vid_id = text_dict['vid_id']
        # vid_dict = self.vid_dict[vid_id]

        # # --- OPTIMIZED VIDEO LOADING ---
        # if self.cache_ram and vid_id in self.video_cache:
        #     # 1. Fast Path: RAM
        #     vid_feats, orig_vid_len = self.video_cache[vid_id]
            
        #     # 2. Recalculate Stride/Size (Logic copied from BaseDataset)
        #     # Since we already resized, we must manually calculate what the stride WOULD be
        #     clip_size = self.clip_size
        #     clip_stride = self.clip_stride
            
        #     # Calculate the ratio based on original length we stored
        #     if self.to_fixed_len:
        #         clip_size = clip_stride = float(
        #             ((orig_vid_len - 1) * self.clip_stride + self.clip_size) / self.max_vid_len
        #         )
        # else:
        #     # 1. Slow Path: Disk (Fallback)
        #     # This calls the parent logic which loads from .npy and resizes
        #     # We cannot easily use super().__getitem__ because we need to inject 'video_id' later
        #     # So we replicate the logic:
        #     vid_feats = self._load_vid_feats(vid_id)
        #     orig_vid_len = vid_feats.size(1)
            
        #     clip_size, clip_stride = self.clip_size, self.clip_stride
        #     if self.to_fixed_len:
        #         vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
        #         clip_size = clip_stride = float(
        #             ((orig_vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
        #         )

        # # --- COMMON LOGIC (Targets, etc.) ---
        # clip_offset = 0.5 * clip_size / clip_stride

        # segments = np.clip(
        #     text_dict['segment'] * vid_dict['fps'], 
        #     a_min=0, a_max=vid_dict['num_frames']
        # ) / clip_stride - clip_offset
        # segments = torch.from_numpy(
        #     np.ascontiguousarray(segments.astype(np.float32))
        # )

        # # Skip truncation logic for fixed_len mode (it's irrelevant)

        # text_feats = self._load_text_feats(text_id)

        # # External scores (if any)
        # ext_scores = None 
        # # (Skipping ext_score logic for brevity unless you need it, adds 5 lines)

        # return {
        #     'fps': vid_dict['fps'],
        #     'num_frames': vid_dict['num_frames'],
        #     'duration': vid_dict['duration'],
        #     'segment': text_dict['segment'], # Raw Seconds
        #     'clip_size': clip_size,
        #     'clip_stride': clip_stride,
        #     'target': segments,              # Grid Units
        #     'sentence': text_dict["text"],
        #     'vid': vid_feats,
        #     'text': text_feats,
        #     'ext_scores': ext_scores,
        #     'video_id': vid_id,
        #     'text_id': text_id
        # }

# =========================================================
# 2. Collate Function (for Fixed Length 256)
# =========================================================
def snag_fixed_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None

    # Stack Video
    # Ensure (T, C) -> (256, 2048)
    vid_list = []
    for b in batch:
        v = b['vid']
        if v.shape[0] == 2048: v = v.transpose(0, 1)
        vid_list.append(v)
    video_emb = torch.stack(vid_list)
    video_mask = torch.ones((len(batch), 256), dtype=torch.bool)

    # Stack Text
    text_list = []
    for b in batch:
        t = b['text']
        if t.shape[0] == 300: t = t.transpose(0, 1)
        text_list.append(t)
    max_len = max([t.shape[0] for t in text_list])
    padded_text = torch.zeros((len(batch), max_len, 300))
    text_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, t in enumerate(text_list):
        l = t.shape[0]
        padded_text[i, :l] = t
        text_mask[i, :l] = True
        
    # --- FIX TARGET SHAPE ---
    raw_targets = []
    for b in batch:
        t = b['target']
        # If shape is (1, 2), squeeze it to (2,)
        if t.dim() > 1 and t.shape[0] == 1:
            t = t.squeeze(0)
        raw_targets.append(t)
        
    targets = torch.stack(raw_targets) # Now consistently (B, 2)
    
    return {
        "video_emb": video_emb, 
        "video_mask": video_mask,
        "query_tokens": padded_text, 
        "query_mask": text_mask,
        "targets": targets, 
        "video_id": [b.get('video_id') for b in batch],
        "durations": [b.get('duration') for b in batch],
        "text_ids": [b.get('text_id') for b in batch]
    }

def snag_custom_collate(batch, fixed_text_len=16):
    """
    Custom Collate function to return specific dictionary structure for Charades-STA.
    Assumes batch items come from CharadesSnagAdapter (Fixed Length=256).
    """
    batch = [b for b in batch if b is not None]
    if not batch: return None

    MAX_VID_LEN = 256   
    
    video_embs = []
    query_tokens_list = []
    
    video_ids = []
    queries = []
    durations = []
    
    start_secs = []
    end_secs = []
    i0s = []
    i1s = []
    
    for item in batch:
        vid_id = item.get('video_id', 'unknown')
        text = item.get('sentence', '')
        duration = item.get('duration', 1.0)
        
        # --- FIX: ROBUST TARGET HANDLING ---
        # 1. Force flatten to 1D: [[s,e]] -> [s,e] or [s,e] -> [s,e]
        target_grid = item['target'].view(-1) 
        
        # 2. Now safe to access indices
        idx_start = int(round(target_grid[0].item()))
        idx_end = int(round(target_grid[1].item()))
        
        # Clamp to valid range [0, 255]
        idx_start = max(0, min(idx_start, MAX_VID_LEN - 1))
        idx_end = max(0, min(idx_end, MAX_VID_LEN - 1))
        
        # Recover Seconds
        s_sec = (idx_start / float(MAX_VID_LEN)) * duration
        e_sec = (idx_end / float(MAX_VID_LEN)) * duration
        
        # -- Video Processing --
        v = item['vid']
        if v.shape[0] == 2048: v = v.transpose(0, 1)
        video_embs.append(v)
        
        # -- Text Processing --
        t = item['text']
        if t.shape[0] == 300: t = t.transpose(0, 1)
        query_tokens_list.append(t)
        
        video_ids.append(vid_id)
        queries.append(text)
        durations.append(duration)
        
        start_secs.append(s_sec)
        end_secs.append(e_sec)
        i0s.append(idx_start)
        i1s.append(idx_end)

    # --- STACK TENSORS ---
    batch_video_emb = torch.stack(video_embs)
    
    batch_max = max([t.shape[0] for t in query_tokens_list])
    
    # LOGIC CHANGE: Decide the final length
    if fixed_text_len is not None:
        # Force it to be 16 (even if batch_max is only 5)
        # But ensure we don't crash if a sentence somehow slipped through larger than 16
        final_len = max(fixed_text_len, batch_max) 
    else:
        final_len = batch_max
    
    text_dim = query_tokens_list[0].shape[1]
    
    # Initialize with Zeros (Padding)
    batch_query_tokens = torch.zeros((len(batch), final_len, text_dim))
    batch_query_mask = torch.zeros((len(batch), final_len), dtype=torch.bool)
    batch_text_emb = torch.zeros((len(batch), text_dim))

    for i, t in enumerate(query_tokens_list):
        l = t.shape[0]
        # Safety check: Truncate if t is longer than final_len
        # (This handles edge cases where dataset truncation failed)
        if l > final_len:
            t = t[:final_len]
            l = final_len
        batch_query_tokens[i, :l] = t
        batch_query_mask[i, :l] = True
        batch_text_emb[i] = torch.mean(t, dim=0)

    # Create Video Mask (All True because of Fixed Length Resizing)
    # Shape: (Batch, 256)
    batch_video_mask = torch.ones((len(batch), MAX_VID_LEN), dtype=torch.bool)
    return {
        "video_id": video_ids,          
        "query": queries,               
        "start_sec": torch.tensor(start_secs, dtype=torch.float32), 
        "end_sec": torch.tensor(end_secs, dtype=torch.float32),
        "i0": torch.tensor(i0s, dtype=torch.long), 
        "i1": torch.tensor(i1s, dtype=torch.long),
        "video_emb": batch_video_emb,   
        "video_mask": batch_video_mask,     # (B, 256) -> All True
        "text_emb": batch_text_emb,     
        "query_tokens": batch_query_tokens, 
        "query_mask": batch_query_mask,     
        "duration": torch.tensor(durations, dtype=torch.float32),
        "stride": 4,                    
        "max_vid_len": MAX_VID_LEN      
    }
def worker_init_fn(worker_id, seed=123):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_dataset(args, batch_size=16):
    print("Loading Dataset...")
    glove_tokenizer = GloVeTokenizer()
    paths = {
    "anno_root": f"{args.data_root}/charades_sta/annotations",
    "vid_feat_dir" : [f"{args.data_root}/charades_sta/i3d_features/charades/rgb", f"{args.data_root}/charades_sta/i3d_features/charades/flow"]
    }

    # Standard SnAG Configuration for Fair Comparison
    train_dataset_cfg = {
        "split": "train",
        "max_vid_len": 256,      # Fixed length
        "to_fixed_len": True,    # Force resizing
        "clip_size": 16,         # These values depend on your feature extraction
        "clip_stride": 4,
        "max_text_len": 16,
        
    }

    # Standard SnAG Configuration for Fair Comparison
    val_dataset_cfg = {
        "split": "val",
        "max_vid_len": 256,      # Fixed length
        "to_fixed_len": True,    # Force resizing
        "clip_size": 16,         # These values depend on your feature extraction
        "clip_stride": 4,
        "max_text_len": 16,
        
    }

    # Standard SnAG Configuration for Fair Comparison
    test_dataset_cfg = {
        "split": "test",
        "max_vid_len": 256,      # Fixed length
        "to_fixed_len": True,    # Force resizing
        "clip_size": 16,         # These values depend on your feature extraction
        "clip_stride": 4,
        "max_text_len": 16,
        
    }

    train_ds = CharadesSnagAdapter(
        is_training=True,
        tokenizer=glove_tokenizer,
        anno_file=f"{paths['anno_root']}/charades_sta_train_split.json",
        vid_feat_dir=paths["vid_feat_dir"], 
        **train_dataset_cfg
        
    )
    val_ds = CharadesSnagAdapter(
        is_training=False,
        tokenizer=glove_tokenizer,
        anno_file=f"{paths['anno_root']}/charades_sta_val_split.json",
        vid_feat_dir=paths["vid_feat_dir"], 
        **val_dataset_cfg
        
    )
    test_ds = CharadesSnagAdapter(
        is_training=False,
        tokenizer=glove_tokenizer,
        anno_file=f"{paths['anno_root']}/charades_sta_test_split.json",
        vid_feat_dir=paths["vid_feat_dir"], 
        **test_dataset_cfg
    )
        

    # Create Loader
    tr_dataloader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        collate_fn=snag_custom_collate, 
        worker_init_fn=worker_init_fn,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        collate_fn=snag_custom_collate, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        collate_fn=snag_custom_collate, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"{len(train_ds)}")
    return train_ds, val_ds, test_ds, tr_dataloader, val_dataloader, test_dataloader


def split_charadessta_train_val(
    train_json_path: str,
    out_train_path: str,
    out_val_path: str,
    out_test_path: str,
    val_ratio: float = 0.1,
    seed: int = 123,
):
    """
    Splits a Charades-STA formatted JSON (nested dict) into Train and Val sets.
    Input Format: {"train": {"VID1": {...}, ...}, "test": {...}}
    Output Format: {"VID1": {...}, "VID2": {...}} (Direct dictionary for Loader)
    """
    random.seed(seed)
    
    print(f"Loading data from {train_json_path}...")
    with open(train_json_path, "r") as f:
        full_data = json.load(f)

    # 1. Access the 'train' split specifically
    if "train" not in full_data:
        raise ValueError(f"Input file must contain a 'train' key. Found: {full_data.keys()}")
    
    # train_data is: {"VID": {"duration": X, "annotations": [...]}, ...}
    train_data = full_data["train"]
    test_data = full_data["test"]
    
    # 2. Get Video IDs and Shuffle
    vids = list(train_data.keys())
    random.shuffle(vids)

    # 3. Calculate Split Index
    n_val = max(1, int(len(vids) * val_ratio))
    val_vids = set(vids[:n_val])
    train_vids = set(vids[n_val:])

    # 4. Create New Dictionaries
    # We preserve the exact structure of the value (duration, annotations, etc.)
    new_train_split = {"train" : {vid: train_data[vid] for vid in train_vids}}
    new_val_split = {"val" : {vid: train_data[vid] for vid in val_vids}}
    test_split = {"test" : test_data}

    # 5. Save outputs
    # We save them as flat dictionaries of videos so the Dataloader can iterate .items()
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_val_path), exist_ok=True)

    with open(out_train_path, "w") as f:
        json.dump(new_train_split, f, indent=2)
        
    with open(out_val_path, "w") as f:
        json.dump(new_val_split, f, indent=2)

    with open(out_test_path, "w") as f:
        json.dump(test_split, f, indent=2)

    # 6. Statistics
    # Calculate sample counts by summing annotation lists
    n_train_samples = sum(len(v["annotations"]) for v in new_train_split["train"].values())
    n_val_samples = sum(len(v["annotations"]) for v in new_val_split["val"].values())
    n_test_samples = sum(len(v["annotations"]) for v in test_split["test"].values())

    print(f"--- Split Complete ---")
    print(f"Original Train Videos: {len(vids)}")
    print(f"New Train: {len(new_train_split['train'])} videos ({n_train_samples} queries) -> Saved to {out_train_path}")
    print(f"New Val:   {len(new_val_split['val'])} videos ({n_val_samples} queries) -> Saved to {out_val_path}")
    print(f"Test:   {len(test_split['test'])} videos ({n_test_samples} queries) -> Saved to {out_test_path}")

