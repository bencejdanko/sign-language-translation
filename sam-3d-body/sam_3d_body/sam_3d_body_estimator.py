# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Dict, List, Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch, prepare_batch_multi
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    @torch.no_grad()
    def process_batch_parallel(
        self,
        imgs: List[Union[str, np.ndarray]],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        masks: Optional[List[Optional[np.ndarray]]] = None,
        cam_ints: Optional[List[Optional[np.ndarray]]] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of N images IN PARALLEL on the GPU.
        This assumes EXACTLY 1 person per image (e.g. How2Sign dataset).
        """
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        n = len(imgs)
        # Parse images
        img_arrays = []
        for i in range(n):
            img = imgs[i]
            if type(img) == str:
                img = load_image(img, backend="cv2", image_format="bgr")
            if type(imgs[i]) == str or len(img.shape) == 3 and img.shape[2] == 3: # Handle bgr input
                if type(imgs[i]) == str:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_arrays.append(img)
            
        height, width = img_arrays[0].shape[:2]
        
        # Parse boxes
        boxes_list = []
        for i in range(n):
            bbox = bboxes[i] if bboxes is not None and bboxes[i] is not None else None
            if bbox is not None:
                boxes_list.append(bbox.reshape(1, 4))
            else:
                boxes_list.append(np.array([0, 0, width, height]).reshape(1, 4))
                
        # Parse masks
        _masks_list = masks if masks is not None else [None] * n
        _masks_score_list = [np.ones(1) if m is not None else None for m in _masks_list]
        
        # Parse cam ints
        _cam_ints = cam_ints if cam_ints is not None else [None] * n
        
        # Build multi-image batched tensors
        batch = prepare_batch_multi(img_arrays, self.transform, boxes_list, _masks_list, _masks_score_list, _cam_ints)
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)
        
        # Run parallel inference
        outputs = self.model.run_inference_batch(
            img_arrays,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs
            
        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[0]):
            res = {
                "bbox": batch["bbox"][idx, 0].cpu().numpy(),
                "focal_length": out["focal_length"][idx],
                "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                "pred_vertices": out["pred_vertices"][idx],
                "pred_cam_t": out["pred_cam_t"][idx],
                "pred_pose_raw": out["pred_pose_raw"][idx],
                "global_rot": out["global_rot"][idx],
                "body_pose_params": out["body_pose"][idx],
                "hand_pose_params": out["hand"][idx],
                "scale_params": out["scale"][idx],
                "shape_params": out["shape"][idx],
                "expr_params": out["face"][idx],
                "mask": _masks_list[idx] if _masks_list is not None else None,
                "pred_joint_coords": out["pred_joint_coords"][idx],
                "pred_global_rots": out["joint_global_rots"][idx],
                "mhr_model_params": out["mhr_model_params"][idx],
            }
            
            if inference_type == "full":
                res["lhand_bbox"] = np.array([
                    (batch_lhand["bbox_center"].flatten(0, 1)[idx][0] - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                    (batch_lhand["bbox_center"].flatten(0, 1)[idx][1] - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                    (batch_lhand["bbox_center"].flatten(0, 1)[idx][0] + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                    (batch_lhand["bbox_center"].flatten(0, 1)[idx][1] + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                ])
                res["rhand_bbox"] = np.array([
                    (batch_rhand["bbox_center"].flatten(0, 1)[idx][0] - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                    (batch_rhand["bbox_center"].flatten(0, 1)[idx][1] - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                    (batch_rhand["bbox_center"].flatten(0, 1)[idx][0] + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                    (batch_rhand["bbox_center"].flatten(0, 1)[idx][1] + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                ])
            
            all_out.append([res])
            
        return all_out


    @torch.no_grad()
    def process_batch(
        self,
        imgs: List[Union[str, np.ndarray]],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        masks: Optional[List[Optional[np.ndarray]]] = None,
        cam_ints: Optional[List[Optional[np.ndarray]]] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of images through SAM 3D Body estimation.

        The model internally processes one image at a time (GPU inference is
        serial due to varying numbers of detected people and differing image
        sizes), but this method provides a clean batching interface that:

        - Accepts a list of images together with optional per-image bboxes,
          masks, and camera intrinsics.
        - Returns one result list per input image, where each element of that
          inner list is a per-person dict identical to what ``process_one_image``
          returns.

        This is the entry point intended for use inside a training or
        preprocessing pipeline, e.g. as the body of a DataLoader worker or
        inside a ``torch.utils.data.Dataset.__getitem__``.

        Args:
            imgs: List of images.  Each entry is either a file-path (str) or
                an RGB numpy array (H x W x 3, uint8).
            bboxes: Optional list of per-image bounding-box arrays, each with
                shape (N, 4) in xyxy format.  Pass ``None`` for a given image
                to fall back to the detector (or full-image box).
            masks: Optional list of per-image mask arrays, each with shape
                (N, H, W) or (N, H, W, 1).  Pass ``None`` to skip masking for
                a given image.
            cam_ints: Optional list of per-image camera-intrinsic tensors,
                each with shape (3, 3).  Pass ``None`` to use the default FOV
                estimator or the default intrinsics.
            det_cat_id: Detection category ID (passed through to detector).
            bbox_thr: Bounding-box confidence threshold for the detector.
            nms_thr: NMS IoU threshold for the detector.
            use_mask: Whether to run SAM2 segmentation when no external mask
                is provided.
            inference_type: One of ``"full"`` (body + hand decoders),
                ``"body"`` (body decoder only) or ``"hand"`` (hand decoder
                only).

        Returns:
            A list of length ``len(imgs)``.  Each element is the list of
            per-person prediction dicts returned by ``process_one_image``
            for the corresponding image (an empty list if no person is
            detected in that image).

        Example::

            estimator = setup_sam_3d_body(...)

            # Plain list of RGB frames from a video
            frames = [frame1_rgb, frame2_rgb, frame3_rgb]
            batch_results = estimator.process_batch(frames, inference_type="body")

            for img_idx, persons in enumerate(batch_results):
                for person in persons:
                    print(img_idx, person["pred_keypoints_2d"].shape)

            # With pre-computed bboxes (e.g. from a detector run separately)
            boxes = [
                np.array([[100, 50, 400, 600]]),   # one person in frame 0
                np.array([[80, 40, 350, 580],
                          [500, 60, 750, 600]]),   # two people in frame 1
                None,                               # use detector / full image for frame 2
            ]
            batch_results = estimator.process_batch(frames, bboxes=boxes)
        """
        n = len(imgs)

        # Normalise optional per-image lists so we can zip cleanly
        _bboxes   = bboxes   if bboxes   is not None else [None] * n
        _masks    = masks    if masks    is not None else [None] * n
        _cam_ints = cam_ints if cam_ints is not None else [None] * n

        if len(_bboxes) != n or len(_masks) != n or len(_cam_ints) != n:
            raise ValueError(
                "imgs, bboxes, masks, and cam_ints must all have the same length."
            )

        all_results: List[List[Dict[str, Any]]] = []
        for img, bbox, mask, cam_int in zip(imgs, _bboxes, _masks, _cam_ints):
            result = self.process_one_image(
                img,
                bboxes=bbox,
                masks=mask,
                cam_int=cam_int,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                use_mask=use_mask,
                inference_type=inference_type,
            )
            all_results.append(result)

        return all_results

    # ------------------------------------------------------------------
    # DataLoader / pipeline helpers
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collate a list of *single-person* prediction dicts into a batch dict.

        This is designed to be used as ``collate_fn`` for a
        ``torch.utils.data.DataLoader`` where each ``Dataset.__getitem__``
        returns a single per-person result dict (i.e. one element from what
        ``process_one_image`` / ``process_batch`` returns).

        Numpy arrays and Python scalars are stacked into numpy arrays.  Other
        types are collected into plain Python lists.

        Args:
            samples: A list of per-person dicts, all with the same keys.

        Returns:
            A single dict where each value is either a stacked numpy array
            (if all values for that key were numpy arrays or scalars) or a
            plain Python list.

        Example::

            from torch.utils.data import DataLoader

            class SignDataset(torch.utils.data.Dataset):
                def __init__(self, frames, estimator):
                    # Pre-process all frames and flatten into a per-person list
                    self.samples = []
                    results = estimator.process_batch(frames, inference_type="body")
                    for frame_idx, persons in enumerate(results):
                        for person in persons:
                            person["frame_idx"] = frame_idx
                            self.samples.append(person)

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            dataset = SignDataset(frames, estimator)
            loader = DataLoader(
                dataset,
                batch_size=8,
                collate_fn=SAM3DBodyEstimator.collate_fn,
            )
            for batch in loader:
                print(batch["pred_keypoints_3d"].shape)  # (8, 70, 3)
        """
        if not samples:
            return {}

        keys = samples[0].keys()
        out: Dict[str, Any] = {}
        for key in keys:
            values = [s[key] for s in samples]
            # Stack numpy arrays; fall back to list for anything else
            if all(isinstance(v, np.ndarray) for v in values):
                out[key] = np.stack(values, axis=0)
            elif all(isinstance(v, (int, float)) for v in values):
                out[key] = np.array(values)
            else:
                out[key] = values
        return out

    @staticmethod
    def prepare_batch(
        img: np.ndarray,
        transform: "Compose",
        boxes: np.ndarray,
        masks: Optional[np.ndarray] = None,
        masks_score: Optional[np.ndarray] = None,
        cam_int: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Thin wrapper around :func:`sam_3d_body.data.utils.prepare_batch.prepare_batch`.

        Exposed here so pipeline code can build a model-ready batch dict
        without importing deep library internals.

        Args:
            img: RGB image as a numpy array (H x W x 3).
            transform: The ``Compose`` transform pipeline from the estimator
                (use ``estimator.transform`` for body or
                ``estimator.transform_hand`` for hand crops).
            boxes: Bounding boxes with shape (N, 4) in xyxy format.
            masks: Optional mask array with shape (N, H, W, 1).
            masks_score: Optional per-mask confidence scores, shape (N,).
            cam_int: Optional camera intrinsic tensor with shape (1, 3, 3).

        Returns:
            A batch dict ready for ``model._initialize_batch`` and
            ``model.run_inference``.

        Example::

            import numpy as np
            import torch

            img_rgb  = ...  # H x W x 3 numpy array
            boxes    = np.array([[x1, y1, x2, y2]])

            batch = SAM3DBodyEstimator.prepare_batch(
                img_rgb, estimator.transform, boxes
            )
            batch = recursive_to(batch, "cuda")
            estimator.model._initialize_batch(batch)
        """
        return prepare_batch(img, transform, boxes, masks, masks_score, cam_int)
