import numpy as np
import open3d as o3d
from plyfile import PlyData


class GaussianSplatScene:
    def __init__(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        colors: np.ndarray,
        opacity: np.ndarray,
    ) -> None:
        self.positions = positions  # (N, 3)
        self.scales = scales  # (N, 3)
        self.rotations = rotations  # (N, 4)
        self.colors = colors  # (N, 3) uint8
        self.opacity = opacity  # (N,) float32

    @classmethod
    def from_ply(cls, ply_path: str) -> "GaussianSplatScene":
        ply = PlyData.read(ply_path)
        data = ply.elements[0].data

        positions = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(
            np.float32
        )
        scales = np.stack(
            [data["scale_0"], data["scale_1"], data["scale_2"]], axis=1
        ).astype(np.float32)
        rotations = np.stack(
            [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1
        ).astype(np.float32)

        # Color from SH DC term
        colors_linear = np.stack(
            [data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1
        ).astype(np.float32)

        # Normalize per-channel using robust statistics
        mean = np.mean(colors_linear, axis=0, keepdims=True)
        std = np.std(colors_linear, axis=0, keepdims=True) + 1e-6
        colors_normalized = (colors_linear - mean) / std

        # Bring into [0,1] space through tanh-based squashing
        colors_normalized = 0.5 * (np.tanh(colors_normalized) + 1.0)

        # Convert into display-ready uint8
        colors = (colors_normalized * 255.0).astype(np.uint8)

        opacity = data["opacity"].astype(np.float32)

        return cls(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacity=opacity,
        )

    def _camera_basis(
        self, camera_pos, lookat=None, up=np.array([0, 0, 1], dtype=np.float64)
    ):
        if lookat is None:
            lookat = self.positions.mean(axis=0)

        C, T, U = (
            camera_pos.astype(np.float64),
            lookat.astype(np.float64),
            up.astype(np.float64),
        )
        z = (T - C) / np.linalg.norm(T - C)
        x = np.cross(z, U)
        x /= np.linalg.norm(x)
        y = np.cross(x, z)
        return np.vstack([x, y, z]).T, C

    def visualize(self, camera_pos, lookat=None):
        R, C = self._camera_basis(camera_pos, lookat)
        P_cam = (R.T @ (self.positions - C).T).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P_cam)
        pcd.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float32) / 255.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960, height=540)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    scene = GaussianSplatScene.from_ply("inputs/Museume.ply")
    scene.visualize(camera_pos=np.array([0, 0, 2], dtype=np.float64))
