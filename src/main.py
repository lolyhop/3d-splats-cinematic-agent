import argparse
from collections import Counter

import numpy as np

from src.scene_classifier import SceneClassifierCluster
from src.renderer import GaussianSplatScene
from src.outdoor_path_planner import PathPlanner as outdoor_path_planner
from src.indoor_path_planner import PathPlanner as indoor_path_planner
from src.utils import *


def process_indoor(scene: GaussianSplatScene) -> None:
    cleaned = clean_scene(scene, voxel_size=1.0, density_threshold=10)

    pivots = find_pivot_points(cleaned, k=7, visualize=False)
    trajectory = catmull_rom_spline(pivots, n_points_per_segment=300)

    try:
        indoor_path_planner.render_flythrough_lookahead(
            scene=scene,
            path=trajectory,
            voxel_size=0.7,
            output="outputs/Theater.mp4",
            fps=30,
            yolo_model="yolov8n.pt",
            yolo_conf=0.65,
            yolo_device="cpu",
            draw_boxes=True,
        )
    except NameError:
        print(
            "[WARN] PathPlanner.render_flythrough_lookahead() not available. Skipping render step."
        )


def process_outdoor(scene: GaussianSplatScene) -> None:
    cleaned = clean_scene(scene, voxel_size=1.0, density_threshold=10)
    pivots = find_pivot_points(cleaned, k=1, visualize=False)

    try:
        path = outdoor_path_planner.build_path(
            cleaned_scene=cleaned,
            pivots=pivots,
            points_per_turn=300,
            start_radius=1.5,
            end_radius=6.5,
            height_offset=1.3,
            ascent_height=2.5,
            turns=2,
            final_close_radius=2.5,
            final_close_turns=2,
            final_close_points_per_turn=250,
        )

        outdoor_path_planner.render_flythrough(
            scene=scene,
            path=path,
            pivot_point=pivots[0],
            voxel_size=2,
            output="outdoor-drone.mp4",
        )
    except NameError:
        print(
            "[WARN] PathPlanner.build_path/render_flythrough not available. Skipping render step."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process a Gaussian Splat scene (auto-classify and run pipeline)."
    )
    parser.add_argument(
        "--scene_path", type=str, required=True, help="Path to .ply scene file."
    )
    parser.add_argument(
        "--force_type",
        type=str,
        choices=["indoor", "outdoor"],
        default=None,
        help="Force pipeline type instead of auto-classification.",
    )
    args = parser.parse_args()

    scene_path = args.scene_path

    # Load scene
    scene = GaussianSplatScene.from_ply(scene_path)

    if args.force_type is not None:
        scene_type = args.force_type
        n_clusters = None
        print(f"[INFO] Forced scene type: {scene_type}")
    else:
        scene_type, n_clusters = SceneClassifierCluster.classify(scene)
        print(
            f"[INFO] Auto-classified scene as '{scene_type}' (clusters = {n_clusters})"
        )

    # Dispatch to pipeline
    if scene_type == "indoor":
        process_indoor(scene)
    elif scene_type == "outdoor":
        process_outdoor(scene)
    else:
        raise ValueError(f"Unknown scene type: {scene_type}")


if __name__ == "__main__":
    main()
