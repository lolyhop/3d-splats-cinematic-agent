import numpy as np
from collections import Counter
from src.renderer import GaussianSplatScene


class SceneClassifier:
    @staticmethod
    def classify(scene: GaussianSplatScene):
        positions = scene.positions
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        extent = maxs - mins
        dx, dy, dz = extent
        area_xy = dx * dy
        max_to_min_ratio = np.max(extent) / (np.min(extent) + 1e-6)

        # Voxel density
        voxel_size = 1.0
        coords = np.floor(positions / voxel_size).astype(int)
        counts = np.array(list(Counter(map(tuple, coords)).values()))
        median_density = np.median(counts)

        if area_xy > 5000 or median_density < 5 or dz > 200:
            scene_type = "outdoor"
        elif area_xy < 2000 or median_density > 50 or max_to_min_ratio > 4:
            scene_type = "indoor"
        else:
            scene_type = "indoor" if median_density > 10 else "outdoor"

        return scene_type


if __name__ == "__main__":
    scenes_files = [
        "inputs/ConferenceHall.ply",
        "inputs/Museume.ply",
        "inputs/Theater.ply",
        "inputs/outdoor-drone.ply",
        "inputs/outdoor-street.ply",
    ]

    for file in scenes_files:
        scene = GaussianSplatScene.from_ply(file)
        scene_type = SceneClassifier.classify(scene)
        print(f"{file.split('/')[-1]} → {scene_type.upper()}\n")
