import contextlib
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _pool_embedding(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 4:
        # BCHW -> BC
        return feat.mean(dim=(-2, -1))
    if feat.ndim == 3:
        # BNC -> BC
        return feat.mean(dim=1)
    if feat.ndim == 2:
        # BC -> BC
        return feat
    # Fallback for unexpected shapes
    return feat.flatten(start_dim=1).mean(dim=1, keepdim=True)


def resolve_dinov3_hook_layer(estimator) -> Tuple[str, torch.nn.Module]:
    model = estimator.model
    candidates = [
        "backbone.encoder.norm",
        "backbone.encoder",
        "backbone",
    ]
    modules = dict(model.named_modules())
    for name in candidates:
        if name in modules:
            return name, modules[name]
    raise RuntimeError(
        "Could not locate a DINOv3 backbone layer to hook. "
        "Expected one of: backbone.encoder.norm, backbone.encoder, backbone."
    )


def print_dinov3_backbone_named_modules(estimator, limit: Optional[int] = None) -> List[str]:
    names = [
        name
        for name, _ in estimator.model.named_modules()
        if name.startswith("backbone")
    ]
    if limit is not None:
        names = names[:limit]
    print("Backbone modules:")
    for name in names:
        print(f"  {name}")
    return names


class PooledBackboneEmbeddingHook(contextlib.AbstractContextManager):
    def __init__(self, estimator, layer_name: Optional[str] = None):
        self.estimator = estimator
        self.layer_name = layer_name
        self._handle = None
        self._batches: Deque[torch.Tensor] = deque()

    def _hook_fn(self, module, inputs, output):
        tensor = _first_tensor(output)
        if tensor is None:
            return
        pooled = _pool_embedding(tensor.detach())
        # NumPy does not support bfloat16, so normalize to float32 at capture time.
        self._batches.append(pooled.float().cpu())

    def pop_batch(self) -> Optional[torch.Tensor]:
        if not self._batches:
            return None
        return self._batches.popleft()

    def __enter__(self):
        if self.layer_name is None:
            self.layer_name, module = resolve_dinov3_hook_layer(self.estimator)
        else:
            modules = dict(self.estimator.model.named_modules())
            if self.layer_name not in modules:
                raise KeyError(f"Layer '{self.layer_name}' not found in estimator.model")
            module = modules[self.layer_name]
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _attach_embeddings_to_batch_results(
    batch_results: List[List[Dict]],
    pooled_batch: Optional[torch.Tensor],
    key_name: str = "pooled_dinov3_embedding",
) -> List[Optional[np.ndarray]]:
    frame_embeddings: List[Optional[np.ndarray]] = []
    if pooled_batch is None:
        for frame_people in batch_results:
            for person in frame_people:
                person[key_name] = None
            frame_embeddings.append(None)
        return frame_embeddings

    n_frames = len(batch_results)
    n_embed = pooled_batch.shape[0]
    n = min(n_frames, n_embed)

    for i in range(n):
        emb_i = pooled_batch[i].float().numpy()
        for person in batch_results[i]:
            person[key_name] = emb_i
        frame_embeddings.append(emb_i)

    for i in range(n, n_frames):
        for person in batch_results[i]:
            person[key_name] = None
        frame_embeddings.append(None)

    return frame_embeddings


def batch_process_video(
    estimator,
    video_path: str,
    batch_size: int = 16,
    inference_type: str = "full",
    hook_layer_name: Optional[str] = None,
):
    """
    Process a video in frame batches and return pose outputs + pooled DINO embeddings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], []

    all_results: List[List[Dict]] = []
    all_embeddings: List[Optional[np.ndarray]] = []
    frames_batch: List[np.ndarray] = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Video")

    with PooledBackboneEmbeddingHook(
        estimator=estimator, layer_name=hook_layer_name
    ) as emb_hook:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if frames_batch:
                        batch_results = estimator.process_batch_parallel(
                            frames_batch, inference_type=inference_type
                        )
                        pooled_batch = emb_hook.pop_batch()
                        frame_embs = _attach_embeddings_to_batch_results(
                            batch_results, pooled_batch
                        )
                        all_results.extend(batch_results)
                        all_embeddings.extend(frame_embs)
                        pbar.update(len(frames_batch))
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)

                if len(frames_batch) == batch_size:
                    batch_results = estimator.process_batch_parallel(
                        frames_batch, inference_type=inference_type
                    )
                    pooled_batch = emb_hook.pop_batch()
                    frame_embs = _attach_embeddings_to_batch_results(
                        batch_results, pooled_batch
                    )
                    all_results.extend(batch_results)
                    all_embeddings.extend(frame_embs)
                    pbar.update(batch_size)
                    frames_batch = []
        finally:
            cap.release()
            pbar.close()

    return all_results, all_embeddings
