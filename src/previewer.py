import streamlit as st
import numpy as np
import open3d as o3d
from plyfile import PlyData
import tempfile

st.title("Standalone Gaussian Splat Scene Preview")

# --- Загрузка PLY ---
ply_file = st.file_uploader("Upload a PLY file", type=["ply"])
if ply_file is not None:
    # Временный файл для Open3D
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
        tmp.write(ply_file.read())
        tmp_path = tmp.name

    # --- Чтение PLY ---
    ply = PlyData.read(tmp_path)
    data = ply.elements[0].data
    positions = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    # Если есть цвета
    if all(name in data.dtype.names for name in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        colors_linear = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)
        # Нормализация
        mean = colors_linear.mean(axis=0, keepdims=True)
        std = colors_linear.std(axis=0, keepdims=True) + 1e-6
        colors_normalized = (colors_linear - mean) / std
        colors_normalized = 0.5 * (np.tanh(colors_normalized) + 1.0)
        colors = (colors_normalized * 255).astype(np.uint8)
    else:
        colors = np.ones_like(positions, dtype=np.uint8) * 127

    # --- Настройка камеры ---
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    camera_x = st.slider("Camera X", float(min_vals[0]), float(max_vals[0]), float((min_vals[0]+max_vals[0])/2))
    camera_y = st.slider("Camera Y", float(min_vals[1]), float(max_vals[1]), float((min_vals[1]+max_vals[1])/2))
    camera_z = st.slider("Camera Z", float(min_vals[2]), float(max_vals[2]), float(max_vals[2]+1.0))
    camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float64)
    lookat = positions.mean(axis=0)

    # --- Вычисляем базис камеры ---
    z = (lookat - camera_pos)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0, 0, 1], dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(x, z)
    R = np.vstack([x, y, z]).T
    P_cam = (R.T @ (positions - camera_pos).T).T

    # --- Создание PointCloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_cam)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(False))
    img = (img * 255).astype(np.uint8)
    vis.destroy_window()

    st.image(img, channels="RGB", use_column_width=True)