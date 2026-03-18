import argparse

from notebook.utils import setup_sam_3d_body

from sam3d_embedding_utils import (
    print_dinov3_backbone_named_modules,
    resolve_dinov3_hook_layer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-repo-id",
        default="facebook/sam-3d-body-dinov3",
        help="HF repo id for SAM-3D-Body weights",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Max number of backbone named_modules to print",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    estimator = setup_sam_3d_body(
        hf_repo_id=args.hf_repo_id,
        detector_name="",
        segmentor_name="",
        fov_name="",
        device=args.device,
    )

    print_dinov3_backbone_named_modules(estimator, limit=args.limit)
    hook_name, hook_module = resolve_dinov3_hook_layer(estimator)
    print("\nRecommended hook layer:")
    print(f"  name: {hook_name}")
    print(f"  module: {hook_module.__class__.__name__}")


if __name__ == "__main__":
    main()
