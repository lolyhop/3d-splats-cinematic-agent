import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import Counter
from src.renderer import GaussianSplatScene


def downsample_positions(positions: np.ndarray, max_points=50000):
    if len(positions) <= max_points:
        return positions
    idx = np.random.choice(len(positions), max_points, replace=False)
    return positions[idx]

import numpy as np
from scipy.interpolate import CubicSpline

def catmull_rom_spline(pivots: np.ndarray, n_points_per_segment: int = 50):
    """
    pivots: (N, 3) - pivot points, N >= 4
    Returns: (M, 3) - smooth trajectory through pivots
    """
    P = np.vstack([pivots[0], pivots, pivots[-1]])  # (N+2, 3)
    trajectory = []

    def CR_point(P0, P1, P2, P3, t):
        """Catmull-Rom formula"""
        t2 = t * t
        t3 = t2 * t
        return (
            0.5
            * (
                (2 * P1)
                + (-P0 + P2) * t
                + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t2
                + (-P0 + 3 * P1 - 3 * P2 + P3) * t3
            )
        )

    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, n_points_per_segment):
            trajectory.append(CR_point(P[i - 1], P[i], P[i + 1], P[i + 2], t))

    return np.array(trajectory, dtype=np.float32)


class PathPlanner:

    @staticmethod
    def clean_scene(
        scene: GaussianSplatScene, voxel_size: float = 0.5, density_threshold: int = 5
    ) -> GaussianSplatScene:
        positions = scene.positions
        coords = np.floor(positions / voxel_size).astype(int)
        counts = Counter(map(tuple, coords))
        keep_mask = np.array([counts[tuple(c)] >= density_threshold for c in coords])
        print(f"Original points: {len(positions)}, after cleaning: {np.sum(keep_mask)}")

        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        extent = maxs - mins

        return GaussianSplatScene(
            positions=positions[keep_mask],
            scales=scene.scales[keep_mask],
            rotations=scene.rotations[keep_mask],
            colors=scene.colors[keep_mask],
            opacity=scene.opacity[keep_mask],
        )

    @staticmethod
    def build_voxel_map(
        scene: GaussianSplatScene,
        voxel_size: float = 1.0,
        density_threshold: int = 10,
        debug: bool = False,
    ):
        """
        Builds a voxel occupancy map and visualizes it using Open3D.
        Each occupied voxel is shown as a cube of size voxel_size.
        """
        coords = np.floor(scene.positions / voxel_size).astype(int)
        counts = Counter(map(tuple, coords))
        occupied_voxels = np.array(
            [np.array(c) for c, n in counts.items() if n >= density_threshold],
            dtype=np.int32,
        )

        print(f"[INFO] Total voxels: {len(counts)}, occupied: {len(occupied_voxels)}")

        if debug is True:
            voxel_centers = (occupied_voxels.astype(np.float32) + 0.5) * voxel_size

            voxel_meshes = []
            cube = o3d.geometry.TriangleMesh.create_box(
                width=voxel_size, height=voxel_size, depth=voxel_size
            )
            cube.compute_vertex_normals()

            for center in voxel_centers:
                cube_copy = o3d.geometry.TriangleMesh(cube)
                cube_copy.translate(center - voxel_size / 2)
                cube_copy.paint_uniform_color([0.3, 0.6, 1.0])  # blue cubes
                voxel_meshes.append(cube_copy)

            full_voxels = voxel_meshes[0]
            for m in voxel_meshes[1:]:
                full_voxels += m

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scene.positions)
            pcd.colors = o3d.utility.Vector3dVector(
                scene.colors.astype(np.float32) / 255.0
            )

            o3d.visualization.draw_geometries([pcd, full_voxels])

        return occupied_voxels
    
    @staticmethod
    def render_flythrough_lookahead(
        scene: "GaussianSplatScene",
        path: np.ndarray,
        voxel_size: float = 1.0,
        output: str = "flythrough_lookahead.mp4",
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

        path_world = path.astype(np.float32)#  * voxel_size

        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        print("[INFO] Rendering flythrough with look-ahead...")

        vertical = 1  # Y-axis
        right_prev = np.array([1, 0, 0], dtype=np.float32)
        R_prev = None
        min_norm = 1e-3
        alpha = 0.08

        for i in tqdm(range(len(path_world) - 1)):
            current = path_world[i]
            next_point = path_world[i + 1]  # динамический look-at

            # --- front ---
            front = next_point - current
            norm_f = np.linalg.norm(front)
            if norm_f < 1e-8:
                front = np.array([0, 0, 1], dtype=np.float32)
            else:
                front = front / norm_f

            # --- horizontal forward ---
            forward_h = front.copy()
            forward_h[vertical] = 0.0
            fh_norm = np.linalg.norm(forward_h)
            if fh_norm < 1e-6:
                forward_h = right_prev.copy()
            else:
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

            # --- smoothing ---
            if R_prev is None:
                R_sm = R_new
            else:
                R_sm = (1.0 - alpha) * R_prev + alpha * R_new
                U, _, Vt = np.linalg.svd(R_sm)
                R_sm = U @ Vt

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
    def render_flythrough(
        scene: "GaussianSplatScene",
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
    def visualize_pivots(
        scene: "GaussianSplatScene", pivot_points: np.ndarray, cube_size: float = 0.5
    ):
        if pivot_points.ndim != 2 or pivot_points.shape[1] != 3:
            raise ValueError("pivot_points must be of shape (N, 3)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.positions)
        pcd.colors = o3d.utility.Vector3dVector(scene.colors.astype(np.float32) / 255.0)

        base_cube = o3d.geometry.TriangleMesh.create_box(
            width=cube_size, height=cube_size, depth=cube_size
        )
        base_cube.compute_vertex_normals()

        cubes = []
        for pivot in pivot_points:
            cube = o3d.geometry.TriangleMesh(base_cube)
            cube.translate(pivot - cube_size / 2)
            cube.paint_uniform_color([1.0, 0.0, 0.0])  # red
            cubes.append(cube)

        o3d.visualization.draw_geometries([pcd, *cubes])

    @staticmethod
    def visualize_trajectory(
        scene: "GaussianSplatScene",
        trajectory: np.ndarray,
        show_points: bool = True,
        line_color=(0.0, 0.8, 1.0),
        point_size: float = 0.15,
    ):
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("trajectory must be of shape (N, 3)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.positions)
        pcd.colors = o3d.utility.Vector3dVector(scene.colors.astype(np.float32) / 255.0)

        geoms = [pcd]

        lines = []
        for i in range(len(trajectory) - 1):
            lines.append([i, i + 1])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(line_color, dtype=np.float32), (len(lines), 1))
        )

        geoms.append(line_set)

        if show_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=point_size)
            sphere.compute_vertex_normals()

            for pt in trajectory:
                sp = o3d.geometry.TriangleMesh(sphere)
                sp.translate(pt)
                sp.paint_uniform_color(line_color)
                geoms.append(sp)

        o3d.visualization.draw_geometries(geoms)

    # TODO: Try KNN, DBSCAN, etc.
    @staticmethod
    def find_pivot_points(
        scene: "GaussianSplatScene",
        voxel_size: float = 1.0,
        num_pivots: int = 3,
        min_distance: float = 3.0,
        density_weight: float = 1.0,
        distance_weight: float = 1.5,
        visualize: bool = True,
    ) -> np.ndarray:
        positions = scene.positions
        if len(positions) == 0:
            raise ValueError("Scene is empty — no positions available.")

        coords = np.floor(positions / voxel_size).astype(int)
        counts = Counter(map(tuple, coords))

        voxel_coords = np.array(list(counts.keys()))
        voxel_densities = np.array(list(counts.values()), dtype=np.float32)
        density_norm = (voxel_densities - voxel_densities.min()) / (
            voxel_densities.max() - voxel_densities.min() + 1e-6
        )

        first_idx = np.argmax(voxel_densities)
        pivots = [voxel_coords[first_idx]]
        remaining_indices = np.delete(np.arange(len(voxel_coords)), first_idx)

        while len(pivots) < num_pivots and len(remaining_indices) > 0:
            pivot_positions = (np.array(pivots) + 0.5) * voxel_size
            voxel_positions = (voxel_coords[remaining_indices] + 0.5) * voxel_size

            dists = np.linalg.norm(
                voxel_positions[:, None, :] - pivot_positions[None, :, :], axis=-1
            )
            min_dists = np.min(dists, axis=1)

            score = density_weight * density_norm[
                remaining_indices
            ] + distance_weight * (min_dists / (min_distance + 1e-6))

            best_idx_local = np.argmax(score)
            best_global_idx = remaining_indices[best_idx_local]

            if min_dists[best_idx_local] < min_distance:
                remaining_indices = np.delete(remaining_indices, best_idx_local)
                continue

            pivots.append(voxel_coords[best_global_idx])
            remaining_indices = np.delete(remaining_indices, best_idx_local)

        pivot_points = (np.array(pivots, dtype=np.float32) + 0.5) * voxel_size

        print(
            f"[INFO] Found {len(pivot_points)} diverse pivots (voxel_size={voxel_size})"
        )

        if visualize:
            PathPlanner.visualize_pivots(
                scene, pivot_points, cube_size=voxel_size * 0.8
            )

        return pivot_points

    @staticmethod
    def find_pivot_points_kmeans(
        scene: GaussianSplatScene,
        k: int = 4,
        pca_enabled: bool = True,
        visualize: bool = True,
    ) -> np.ndarray:
        positions = scene.positions.astype(np.float32)
        pts = downsample_positions(positions, max_points=60000)

        if pca_enabled:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=3)
            pts_pca = pca.fit_transform(pts)

            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=10)
            km.fit(pts_pca)
            centers = pca.inverse_transform(km.cluster_centers_)
        else:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=10)
            km.fit(pts)
            centers = km.cluster_centers_

        centers = centers.astype(np.float32)

        if visualize:
            PathPlanner.visualize_pivots(scene, centers)

        return centers

    @staticmethod
    def build_outdoor_path(
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


def auto_voxel_params(scene, voxel_size=0.5, percentile=60):
    coords = np.floor(scene.positions / voxel_size).astype(int)
    counts = np.array(list(Counter(map(tuple, coords)).values()))
    density_threshold = max(1, int(np.percentile(counts, percentile)))
    print(f"Auto voxel_size: {voxel_size:.2f}, density_threshold: {density_threshold}")
    return voxel_size, density_threshold


if __name__ == "__main__":

    scene = GaussianSplatScene.from_ply("inputs/Theater.ply")
    cleaned_scene = PathPlanner.clean_scene(scene, voxel_size=1.0, density_threshold=10)
    # pivots = PathPlanner.find_pivot_points(
    #     cleaned_scene, voxel_size=0.5, num_pivots=4, min_distance=10, visualize=True
    # )
    pivots = PathPlanner.find_pivot_points_kmeans(cleaned_scene, k=7, visualize=False)
    trajectory = catmull_rom_spline(pivots, n_points_per_segment=130)

    # PathPlanner.visualize_trajectory(scene, trajectory)

    # path = PathPlanner.build_outdoor_path(
    #     cleaned_scene,
    #     pivots,
    #     points_per_turn=300,
    #     start_radius=1.5,
    #     end_radius=6.5,
    #     height_offset=1.3,
    #     ascent_height=2.5,
    #     turns=2,
    #     final_close_radius=2.5,
    #     final_close_turns=2,
    #     final_close_points_per_turn=250,
    # )

    # # PathPlanner.visualize_trajectory(cleaned_scene, path)

    # PathPlanner.render_flythrough(
    #     scene,
    #     path,
    #     pivot_point=pivots[0],
    #     voxel_size=2,
    #     output="outdoor_drone.mp4",
    # )

    PathPlanner.render_flythrough_lookahead(
        scene,
        trajectory,
        voxel_size=2,
        output="Theater.mp4",
    )