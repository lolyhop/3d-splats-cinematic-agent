import numpy as np
from sklearn.cluster import DBSCAN
from src.renderer import GaussianSplatScene


class SceneClassifierCluster:
    @staticmethod
    def _voxel_downsample(positions, voxel_size=1.0):
        """Downsample points using voxel grid."""
        coords = np.floor(positions / voxel_size).astype(int)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return positions[indices]

    @staticmethod
    def _subsample_points(positions, max_points=50000):
        """Random subsampling to limit point count."""
        if len(positions) > max_points:
            idx = np.random.choice(len(positions), max_points, replace=False)
            return positions[idx]
        return positions

    @staticmethod
    def classify(
        scene: GaussianSplatScene,
        voxel_size: float = 1.0,
        max_points: int = 50000,
        eps: float = 5.0,
        min_samples: int = 50,
    ):
        positions = scene.positions.astype(np.float32)

        # --- Downsample to reduce points ---
        positions = SceneClassifierCluster._voxel_downsample(positions, voxel_size)
        positions = SceneClassifierCluster._subsample_points(positions, max_points)

        # --- DBSCAN clustering ---
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # --- Heuristic classification based on number of clusters ---
        if n_clusters < 4:
            scene_type = "indoor"
        else:
            scene_type = "outdoor"

        return scene_type, n_clusters


if __name__ == "__main__":
    scene_files = [
        "inputs/ConferenceHall.ply",
        "inputs/Museume.ply",
        "inputs/Theater.ply",
        "inputs/outdoor-drone.ply",
        "inputs/outdoor-street.ply",
    ]

    for file in scene_files:
        scene = GaussianSplatScene.from_ply(file)
        scene_type, n_clusters = SceneClassifierCluster.classify(scene)
        print(f"{file.split('/')[-1]} → {scene_type.upper()}, clusters = {n_clusters}")
