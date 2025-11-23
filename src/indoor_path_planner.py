import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import Counter
from src.renderer import GaussianSplatScene
from src.utils import *


class PathPlanner:

    @staticmethod
    def render_flythrough_lookahead(
        scene: "GaussianSplatScene",
        path: np.ndarray,
        voxel_size: float = 1.0,
        output: str = "flythrough_lookahead.mp4",
        fps: int = 30,
        yolo_model: str = "yolov8n.pt",
        yolo_conf: float = 0.25,
        yolo_device: str = "cpu",
        draw_boxes: bool = True,
    ):
        """
        Renders flythrough with YOLO bounding boxes only for detected frames.
        """
        # Prepare scene
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
        path_world = path.astype(np.float32)

        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        print("[INFO] Rendering flythrough with YOLO...")

        # Load YOLO
        yolo_model_obj = None
        yolo_names = None
        try:
            from ultralytics import YOLO

            yolo_model_obj = YOLO(yolo_model)
            if hasattr(yolo_model_obj, "model") and hasattr(
                yolo_model_obj.model, "names"
            ):
                yolo_names = yolo_model_obj.model.names
            elif hasattr(yolo_model_obj, "names"):
                yolo_names = yolo_model_obj.names
        except Exception as e:
            print("[WARN] YOLO disabled:", e)

        vertical = 1
        right_prev = np.array([1, 0, 0], dtype=np.float32)
        R_prev = None
        min_norm = 1e-3
        alpha = 0.08

        # Main loop
        for i in tqdm(range(len(path_world) - 1)):
            current = path_world[i]
            next_point = path_world[i + 1]

            # Camera orientation
            front = next_point - current
            norm_f = np.linalg.norm(front)
            front = (
                front / norm_f
                if norm_f >= 1e-8
                else np.array([0, 0, 1], dtype=np.float32)
            )

            forward_h = front.copy()
            forward_h[vertical] = 0.0
            fh_norm = np.linalg.norm(forward_h)
            forward_h = forward_h / fh_norm if fh_norm >= 1e-6 else right_prev.copy()

            global_up = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(global_up, forward_h)
            norm_r = np.linalg.norm(right)
            right = right / norm_r if norm_r >= min_norm else right_prev.copy()
            right_prev = right.copy()

            up_vec = np.cross(front, right)
            norm_u = np.linalg.norm(up_vec)
            up_vec = (
                up_vec / norm_u
                if norm_u >= 1e-6
                else np.array([0, 0, 1], dtype=np.float32)
            )

            R_new = np.vstack([right, up_vec, front]).astype(np.float64)
            U, _, Vt = np.linalg.svd(R_new)
            R_new = U @ Vt
            R_sm = R_new if R_prev is None else (1 - alpha) * R_prev + alpha * R_new
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
            img_float = np.asarray(vis.capture_screen_float_buffer(False))
            img_rgb = (img_float * 255).astype(np.uint8)

            # YOLO detection
            boxes = []
            confs = []
            clss = []
            if yolo_model_obj is not None:
                try:
                    results = yolo_model_obj.predict(
                        source=img_rgb,
                        conf=yolo_conf,
                        device=yolo_device,
                        verbose=False,
                    )
                    if len(results) > 0:
                        res = results[0]
                        boxes = (
                            res.boxes.xyxy.cpu().numpy()
                            if hasattr(res.boxes, "xyxy")
                            else []
                        )
                        confs = (
                            res.boxes.conf.cpu().numpy()
                            if hasattr(res.boxes, "conf")
                            else []
                        )
                        clss = (
                            res.boxes.cls.cpu().numpy().astype(int)
                            if hasattr(res.boxes, "cls")
                            else []
                        )
                except Exception as e:
                    print(f"[WARN] YOLO inference failed on frame {i}: {e}")

            # Draw boxes
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if draw_boxes and len(boxes) > 0:
                for k, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = float(confs[k]) if k < len(confs) else 0.0
                    cls_id = int(clss[k]) if k < len(clss) else -1
                    label_text = (
                        f"{yolo_names[cls_id]} {conf:.2f}"
                        if yolo_names
                        else f"class {cls_id} {conf:.2f}"
                    )
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        img_bgr,
                        (x1, max(0, y1 - th - 6)),
                        (x1 + tw + 6, y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        img_bgr,
                        label_text,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

            writer.write(img_bgr)

        writer.release()
        vis.destroy_window()
        print(f"[INFO] Video saved: {output}")


if __name__ == "__main__":

    scene = GaussianSplatScene.from_ply("inputs/Theater.ply")
    cleaned_scene = clean_scene(scene, voxel_size=1.0, density_threshold=10)
    pivots = find_pivot_points(cleaned_scene, k=7, visualize=False)
    trajectory = catmull_rom_spline(pivots, n_points_per_segment=300)

    occupied_voxels = build_voxel_map(
        scene, voxel_size=1, density_threshold=10, debug=False
    )

    PathPlanner.render_flythrough_lookahead(
        scene,
        trajectory,
        voxel_size=0.7,
        output="outputs/Theater.mp4",
        fps=30,
        yolo_model="yolov8n.pt",
        yolo_conf=0.65,
        yolo_device="cpu",
        draw_boxes=True,
    )
