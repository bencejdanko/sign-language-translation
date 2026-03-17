# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import default_collate


class NoCollate:
    def __init__(self, data):
        self.data = data


def _normalize_cam_int(cam_int, ref_tensor):
    """Normalize camera intrinsics to shape ``[1, 3, 3]`` on ref dtype/device."""
    if not torch.is_tensor(cam_int):
        cam_int = torch.as_tensor(cam_int)

    cam_int = cam_int.to(ref_tensor)

    # Accept both (3, 3) and (1, 3, 3) inputs.
    if cam_int.ndim == 2:
        if tuple(cam_int.shape) != (3, 3):
            raise ValueError(f"cam_int must be (3, 3) or (1, 3, 3), got {tuple(cam_int.shape)}")
        cam_int = cam_int.unsqueeze(0)
    elif cam_int.ndim == 3:
        if tuple(cam_int.shape[1:]) != (3, 3):
            raise ValueError(f"cam_int must be (3, 3) or (1, 3, 3), got {tuple(cam_int.shape)}")
        if cam_int.shape[0] != 1:
            # Per-image path only supports one intrinsics matrix.
            cam_int = cam_int[:1]
    else:
        raise ValueError(f"cam_int must be (3, 3) or (1, 3, 3), got {tuple(cam_int.shape)}")

    return cam_int


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = _normalize_cam_int(cam_int, batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch


def prepare_batch_multi(
    imgs,
    transform,
    boxes_list,
    masks_list=None,
    masks_score_list=None,
    cam_ints=None,
):
    """Build a batched dict for N images, each with exactly 1 person.

    Unlike :func:`prepare_batch` which handles one image with up to N people,
    this function handles N images each with exactly **one** person crop.
    The resulting batch dict has shape ``[N, 1, ...]`` across all spatial keys,
    which allows the model's GPU forward passes to process all N frames in
    a single call.

    This is the correct batch structure for How2Sign-style data where there is
    always exactly one signer per frame.

    Args:
        imgs: List of N RGB numpy arrays, each shape (H, W, 3).
            All images must be the same spatial resolution.
        transform: The ``Compose`` transform pipeline (e.g.
            ``estimator.transform`` for body crops).
        boxes_list: List of N arrays, each shape (1, 4) in xyxy format.
            Exactly one bounding box per image.
        masks_list: Optional list of N mask arrays, each shape (1, H, W, 1)
            or ``None`` to use a zero mask for that image.
        masks_score_list: Optional list of N scalar arrays, one per image.
        cam_ints: Optional list of N camera-intrinsic tensors, each (1, 3, 3),
            or ``None`` to use default intrinsics derived from image size.

    Returns:
        A batch dict with spatial keys having shape ``[N, 1, ...]``.
        This can be passed directly to
        ``model._initialize_batch`` → ``model.run_inference_batch``.
    """
    n = len(imgs)
    assert len(boxes_list) == n, "imgs and boxes_list must have the same length"

    _masks_list = masks_list if masks_list is not None else [None] * n
    _masks_score_list = masks_score_list if masks_score_list is not None else [None] * n
    _cam_ints = cam_ints if cam_ints is not None else [None] * n

    # Build one single-person batch per image, then stack along dim-0.
    # Each per-image batch has spatial keys shaped [1, 1, ...]; after stacking
    # they become [N, 1, ...].
    per_image_batches = []
    for img, boxes, masks, masks_score, cam_int in zip(
        imgs, boxes_list, _masks_list, _masks_score_list, _cam_ints
    ):
        # boxes must have shape (1, 4) - exactly one person
        boxes = np.asarray(boxes).reshape(1, 4)
        b = prepare_batch(img, transform, boxes, masks, masks_score, cam_int)
        per_image_batches.append(b)

    stacked: dict = {}
    tensor_keys = [
        "img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
        "bbox", "affine_trans", "mask", "mask_score", "person_valid", "cam_int",
    ]
    for key in tensor_keys:
        if key in per_image_batches[0]:
            stacked[key] = torch.cat([b[key] for b in per_image_batches], dim=0)

    # img_ori is a list of NoCollate wrappers - keep as list of length N
    stacked["img_ori"] = [b["img_ori"][0] for b in per_image_batches]

    return stacked
