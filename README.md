# Sign Language Translation

![alt text](image.png)

Video-to-sign language translation using [facebook/sam-3d-body](https://github.com/facebookresearch/sam-3d-body) and the [How2Sign](https://how2sign.github.io/) dataset.

## Custom SAM-3D-Body Fork

We've modified the core `sam-3d-body` library to natively support parallel bounding-box and keypoint inference across multiple video frames entirely on the GPU.

Since the How2Sign dataset consistently features exactly **1 person per frame**, we can structure batch dicts as `[N_frames, 1, C, H, W]` to process continuous video streams efficiently.

### Using Parallel Batch Inference

You can run true parallel inference across clips by utilizing the new `process_batch_parallel` pipeline entry point. 

```python
from sam_3d_body import SAM3DBodyEstimator
import cv2

# Initialize your estimator wrapper
estimator = setup_sam_3d_body(...)

# Load a contiguous chunk of N frames from your How2Sign dataset
frames = [cv2.imread(f"frame_{i}.jpg") for i in range(16)]

# Run parallel GPU inference
# (This distributes the single-person frames as a hardware batch)
batch_results = estimator.process_batch_parallel(frames, inference_type="full")

for frame_idx, person_stats in enumerate(batch_results):
    # person_stats is a length-1 list for this frame
    pose_3d = person_stats[0]["pred_keypoints_3d"]
    print(f"Frame {frame_idx} 3D Keypoints: {pose_3d.shape}")
```