import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import Counter
from src.renderer import GaussianSplatScene
from src.utils import *


class PathPlanner:

    @staticmethod
    def render_flythrough(
        scene: GaussianSplatScene,
        path: np.ndarray,
        pivot_point: np.ndarray,
        voxel_size: float = 1.0,
        output: str = "flythrough.mp4",
        fps: int = 30,
    ):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.positions)
        pcd.colors = o3d.utility.Vector3dVector(scene.colors.astype(np.float32) / 255.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720, visible=False)
        vis.add_geometry(pcd)

        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0, 0, 0])
        render_opt.point_size = 0.5

        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()

        path_world = path.astype(np.float32) * voxel_size

        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        print("[INFO] Rendering flythrough...")

        vertical = 1  # Y-axis
        right_prev = np.array([1, 0, 0], dtype=np.float32)
        R_prev = None
        min_norm = 1e-3
        alpha = 0.08

        for i in tqdm(range(len(path_world) - 1)):
            current = path_world[i]

            # --- front ---
            front = pivot_point - current
            norm_f = np.linalg.norm(front)
            if norm_f < 1e-8:
                print(f"[WARN][{i}] front too small: {norm_f}")
                continue
            front = front / norm_f

            forward_h = front.copy()
            forward_h[vertical] = 0.0
            fh_norm = np.linalg.norm(forward_h)
            if fh_norm < 1e-6:
                forward_h = right_prev.copy()
                fh_norm = np.linalg.norm(forward_h)
            forward_h /= fh_norm

            # --- right ---
            global_up = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(global_up, forward_h)
            norm_r = np.linalg.norm(right)
            if norm_r < min_norm:
                right = right_prev.copy()
            else:
                right /= norm_r
                right_prev = right.copy()

            # --- up ---
            up_vec = np.cross(front, right)
            norm_u = np.linalg.norm(up_vec)
            if norm_u < 1e-6:
                up_vec = np.array([0, 0, 1], dtype=np.float32)
            else:
                up_vec /= norm_u

            # --- Rotation matrix ---
            R_new = np.vstack([right, up_vec, front]).astype(np.float64)
            U, _, Vt = np.linalg.svd(R_new)
            R_new = U @ Vt

            # --- Smoothing ---
            if R_prev is None:
                R_sm = R_new
            else:
                R_sm = (1.0 - alpha) * R_prev + alpha * R_new
                U, _, Vt = np.linalg.svd(R_sm)
                R_sm = U @ Vt

            if R_prev is not None:
                delta = np.linalg.norm(R_sm - R_prev)
                if delta > 1.0:
                    print(f"[WARN][{i}] sudden orientation jump: {delta:.4f}")
            R_prev = R_sm.copy()

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R_sm
            extrinsic[:3, 3] = -R_sm @ current
            cam_params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(cam_params)

            vis.poll_events()
            vis.update_renderer()
            img = np.asarray(vis.capture_screen_float_buffer(False))
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img)

        writer.release()
        vis.destroy_window()
        print(f"[INFO] Video saved: {output}")

    @staticmethod
    def build_path(
        scene: "GaussianSplatScene",
        pivot_points: np.ndarray,
        start_radius: float = 0.0,
        end_radius: float = 20.0,
        height_offset: float = 5.0,
        ascent_height: float = 30.0,
        turns: int = 3,
        points_per_turn: int = 60,
        final_close_radius: float = 1.0,
        final_close_turns: int = 2,
        final_close_points_per_turn: int = 60,
    ) -> np.ndarray:
        """
        Builds an upward spiral path expanding outward from pivot.
        At the end, adds approach to pivot + small close circles.
        """
        if pivot_points is None or len(pivot_points) == 0:
            raise ValueError("No pivot points provided.")

        pivot = np.array(pivot_points[0], dtype=np.float32)
        vertical = 1  # Y-axis vertical
        horiz_axes = [0, 1, 2]
        horiz_axes.remove(vertical)
        h1, h2 = horiz_axes

        v_mean = scene.positions[:, vertical].mean()
        direction = -1.0 if pivot[vertical] < v_mean else 1.0

        total_points = turns * points_per_turn + 1
        angles = np.linspace(0, 2 * np.pi * turns, total_points, dtype=np.float32)
        heights = np.linspace(
            pivot[vertical] + direction * height_offset,
            pivot[vertical] + direction * (height_offset + ascent_height),
            total_points,
            dtype=np.float32,
        )
        radii = np.linspace(start_radius, end_radius, total_points, dtype=np.float32)

        path = np.zeros((total_points, 3), dtype=np.float32)
        path[:, h1] = pivot[h1] + radii * np.cos(angles)
        path[:, h2] = pivot[h2] + radii * np.sin(angles)
        path[:, vertical] = heights

        def bezier_quad(p0, p1, p2, n_points):
            t = np.linspace(0, 1, n_points)[:, None]
            return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

        approach_points = final_close_points_per_turn
        last_pos = path[-1]

        ctrl_offset = np.array(
            [final_close_radius, final_close_radius, height_offset / 2],
            dtype=np.float32,
        )
        ctrl_pos = last_pos + ctrl_offset * direction

        target_pos = pivot.copy()
        target_pos[vertical] += direction * height_offset

        approach_points = final_close_points_per_turn
        last_pos = path[-1]
        target_pos = pivot.copy()
        target_pos[vertical] += direction * height_offset

        ctrl_offset = np.array(
            [final_close_radius, final_close_radius, height_offset / 2],
            dtype=np.float32,
        )
        ctrl_pos = last_pos + ctrl_offset * direction

        def ease_in_out(t):
            return t**2 * (3 - 2 * t)

        # ease-in/ease-out
        t = np.linspace(0, 1, approach_points)
        t_eased = ease_in_out(t)
        approach = bezier_quad(last_pos, ctrl_pos, target_pos, approach_points)
        for i in range(approach_points):
            approach[i] = (
                last_pos * (1 - t_eased[i])
                + target_pos * t_eased[i]
                + np.array([np.sin(t_eased[i] * 2 * np.pi * 3) * 0.7, 0, 0])
            )

        close_total_points = final_close_turns * final_close_points_per_turn + 1
        close_path = np.zeros((close_total_points, 3), dtype=np.float32)
        start_pos = approach[-1]
        start_radius = np.linalg.norm(start_pos[[h1, h2]] - pivot[[h1, h2]])

        close_path[0] = start_pos
        vel = approach[-1] - approach[-2]

        vel_h = vel[[h1, h2]]

        if np.linalg.norm(vel_h) < 1e-6:
            start_angle = np.arctan2(
                start_pos[h2] - pivot[h2], start_pos[h1] - pivot[h1]
            )
        else:
            start_angle = np.arctan2(vel_h[1], vel_h[0])

        start_pos = approach[-1]

        start_radius = np.linalg.norm(start_pos[[h1, h2]] - pivot[[h1, h2]])

        for i in range(1, close_total_points):
            t = i / (close_total_points - 1)

            radius = (1 - t) * start_radius + t * final_close_radius

            angle = start_angle + t * (2 * np.pi * final_close_turns)

            close_path[i, h1] = pivot[h1] + radius * np.cos(angle)
            close_path[i, h2] = pivot[h2] + radius * np.sin(angle)
            close_path[i, vertical] = start_pos[vertical]

        full_path = np.concatenate([path, approach, close_path], axis=0)
        print(
            f"[build_expanding_spiral_path] pivot = {pivot}, full path shape = {full_path.shape}"
        )
        return full_path


if __name__ == "__main__":

    scene = GaussianSplatScene.from_ply("inputs/outdoor-drone.ply")
    cleaned_scene = clean_scene(scene, voxel_size=1.0, density_threshold=10)
    pivots = find_pivot_points(cleaned_scene, k=1, visualize=False)

    path = PathPlanner.build_path(
        cleaned_scene,
        pivots,
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

    PathPlanner.render_flythrough(
        scene,
        path,
        pivot_point=pivots[0],
        voxel_size=2,
        output="outdoor-drone.mp4",
    )
