import numpy as np
from collections import Counter
from src.renderer import GaussianSplatScene
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def downsample_positions(positions: np.ndarray, max_points=50000):
    if len(positions) <= max_points:
        return positions
    idx = np.random.choice(len(positions), max_points, replace=False)
    return positions[idx]


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
        return 0.5 * (
            (2 * P1)
            + (-P0 + P2) * t
            + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t2
            + (-P0 + 3 * P1 - 3 * P2 + P3) * t3
        )

    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, n_points_per_segment):
            trajectory.append(CR_point(P[i - 1], P[i], P[i + 1], P[i + 2], t))

    return np.array(trajectory, dtype=np.float32)


def clean_scene(
    scene: GaussianSplatScene, voxel_size: float = 0.5, density_threshold: int = 5
) -> GaussianSplatScene:
    positions = scene.positions
    coords = np.floor(positions / voxel_size).astype(int)
    counts = Counter(map(tuple, coords))
    keep_mask = np.array([counts[tuple(c)] >= density_threshold for c in coords])
    print(f"Original points: {len(positions)}, after cleaning: {np.sum(keep_mask)}")

    return GaussianSplatScene(
        positions=positions[keep_mask],
        scales=scene.scales[keep_mask],
        rotations=scene.rotations[keep_mask],
        colors=scene.colors[keep_mask],
        opacity=scene.opacity[keep_mask],
    )


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


def visualize_trajectory(
    scene: GaussianSplatScene,
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


def find_pivot_points(
    scene: GaussianSplatScene,
    k: int = 4,
    visualize: bool = True,
) -> np.ndarray:
    positions = scene.positions.astype(np.float32)
    pts = downsample_positions(positions, max_points=60000)

    pca = PCA(n_components=3)
    pts_pca = pca.fit_transform(pts)
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(pts_pca)
    centers = pca.inverse_transform(km.cluster_centers_)
    centers = centers.astype(np.float32)

    if visualize:
        visualize_pivots(scene, centers)

    return centers


def build_voxel_map(
    scene: GaussianSplatScene,
    voxel_size: float = 1.0,
    density_threshold: int = 10,
    debug: bool = False,
):
    """
    Builds a voxel occupancy map and visualizes it using Open3D.
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
        pcd.colors = o3d.utility.Vector3dVector(scene.colors.astype(np.float32) / 255.0)

        o3d.visualization.draw_geometries([pcd, full_voxels])

    return occupied_voxels
