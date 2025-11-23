# 3D Splats Cinematic Agent Technical Report

# Completed Tasks
1. Render a video from inside the scene. (Theater.mp4 is attached)
2. Detect objects in the rendered video. (Theater.mp4 contains several YOLO predictions)
3. Path planning (both outdoor_drone.mp4 and Theater.mp4 are generated based on specific path building algorithms)
4. Real-time scene preview (scene_preview.mp4 is attached)
5. Rendered video that covers most of the scene/area (Theater.mp4 explores 7 pivot points that cover the most scene area)
6. Render a 360° video (full view of the scene from multiple angles; this is not a true 360° video for platforms like YouTube)

# Approach Overview

To build a cinematic 3D Gaussian splats agent, I split the pipeline into several stages. Each stage solves one part of the problem, and together they allow the agent to fly through a scene and capture cinematic shots.

### 1. Scene Classifier

The first step is to determine what type of scene we are dealing with. Indoor and outdoor environments require different camera paths, so I classify the scene before planning any movement.

### 2. Find Points of Interest

After the scene type is known, the next goal is to find interesting areas in the environment where the camera should fly. These points can be visually dense areas, large structures, or parts of the scene that are significantly different from their surroundings.

### 3. Build a Path Across Selected Points

Once the agent has a list of points of interest, it builds a smooth flight path that connects them. The path generation logic depends on the scene type: indoor paths avoid walls and tight spaces, while outdoor paths are wider and more free-form.

### 4. Object Detection

To better understand what appears in the scene, I run YOLO-based object detection on the rendered RGB frames. The model produces bounding boxes around detected objects, and these boxes can be visualized or highlighted during playback. This helps inspect the scene content and analyze how the agent interacts with different elements.

### 5. Camera Movement

Finally, the agent executes the camera movement along the planned path. The camera orientation is adjusted to keep important points or detected objects in frame.

# Detailed Description

## 1. Scene Classifier

When I started this project, I realized that different types of scenes require different strategies for moving the camera. Indoor spaces, like rooms or castles, have walls and corridors, while outdoor scenes, like parks, are open and do not require careful camera movements with obstacles avoidance. Because of this, I decided to classify the scene first and then use a different path planner depending on the scene type.

After exploring the scenes available on [Superspl.at](https://superspl.at)￼, I chose to divide them into two classes:
- **Indoor scenes:** castles, rooms, houses;
- **Outdoor scenes:** streets, parks, open spaces.

To classify the scenes, I used clustering on the point cloud of each scene. Since the scenes can have hundreds of thousands of points, I first reduced the number of points to make the process faster and more efficient:
1. Points downsampling — combine points that are very close into one;
2. Random subsampling — if there are still too many points, take a random subset up to 50,000 points;
3. DBSCAN clustering — group the points into clusters.

Finally, the number of clusters decides the scene type:
- Few clusters (less than 4) —> indoor scene;
- Many clusters —> outdoor scene;

Here are scene classification results:

| Scene File            | Scene Type | Clusters |
|------------------------|------------|----------|
| ConferenceHall.ply     | Indoor     | 1        |
| Museume.ply            | Indoor     | 1        |
| Theater.ply            | Indoor     | 1        |
| outdoor-drone.ply      | Outdoor    | 13        |
| outdoor-street.ply     | Outdoor    | 11        |

## 2. Find Points of Interest

After the scene type is known, the next step is to detect the most important areas in the environment. These regions later define where the camera should fly and what it should focus on.

### Cleaning Noisy Points

Raw Gaussian splat scenes often contain small noisy clusters or isolated points. They do not represent meaningful geometry and may negatively affect clustering. To make the algorithm more stable, I apply a simple density-based cleaning step. It filters out about 2–5% of points, which is enough to stabilize clustering without changing the structure of the scene.

### Clustering With K-Means

Once the point cloud is cleaned, I search for high-level regions in the scene using clustering algorithm — K-Means. To keep computation efficient, I first downsample the scene to about 60,000 points.

For clustering, I use a combination of PCA and K-Means:
- PCA reduces the point cloud to a space where the global shape is easier to separate;
- K-Means finds the large-scale clusters representing major areas or structures in the environment.

The output of this step is a set of 3D coordinates that represent the main pivot points of the scene.

### Indoor vs Outdoor Logic

Since indoor and outdoor environments have very different structure, the number of points of interest also depends on the scene type:
- **Outdoor scenes:**
I select only one cluster — the largest one. Outdoor areas are open, and a single pivot point is enough to define a central point of interest for the camera.
- **Indoor scenes:**
I select the six largest clusters. This number was chosen empirically and works well for typical indoor environments like rooms, hallways, and multi-part interiors.

These pivot points are then passed to the next stage, where the agent builds a smooth camera path connecting them.

## 3. Build a Path Across Selected Points

Once the pivot points are selected, the next stage is to generate a continuous and cinematic camera trajectory. The exact strategy depends on whether the scene is indoor or outdoor. 

### Outdoor Path Generation

Outdoor scenes typically provide a single dominant pivot point that acts as the visual anchor of the environment. Instead of simply circling around it on a fixed radius, the camera performs a multi-stage motion sequence that resembles a drone shot. 

The path is constructed as a sequence of 3D points in space and consists of three conceptual stages:

**1. Expanding Spiral:**
The camera starts near the pivot and gradually moves outward while ascending. The radius increases linearly, the height changes smoothly, and the angle rotates around the pivot multiple times. This provides an establishing shot of the scene from a dynamic perspective.

```python
# Expanding spiral
spiral_points = []
for i in range(total_points):
    t = i / total_points
    radius = start_radius * (1 - t) + end_radius * t
    height = start_height * (1 - t) + end_height * t
    angle = 2 * np.pi * turns * t
    x = pivot[0] + radius * np.cos(angle)
    y = pivot[1] + radius * np.sin(angle)
    z = pivot[2] + height
    spiral_points.append([x, y, z])
```

**2. Smooth Approach:**
After the spiral, the camera transitions toward the pivot along a smooth Bézier curve. An ease-in/ease-out function modulates the motion to avoid abrupt accelerations, and small oscillations can be added to make the movement feel more natural.

```python
# Quadratic Bézier curve: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
P0 = spiral_points[-1]
P2 = pivot + np.array([0, 0, height_offset])
P1 = P0 + (P2 - P0) * 0.5 + np.array([1.0, 0.5, 0.0])  # control point offset
approach_points = []
for i in range(approach_steps):
    t = i / approach_steps
    t_eased = t**2 * (3 - 2*t)  # ease-in/ease-out
    point = (1 - t_eased)**2 * P0 + 2*(1 - t_eased)*t_eased * P1 + t_eased**2 * P2
    point += np.array([np.sin(t_eased * 6*np.pi) * 0.5, 0, 0])  # small oscillation
    approach_points.append(point)
```

**3. Final Close Orbit:**
Once near the pivot, the camera performs a series of tight circular orbits at gradually decreasing radii.

### Indoor Path Generation

For indoor scenes, multiple pivot points are selected using K-Means algorithm. The goal is to generate a trajectory that moves smoothly between these points, avoiding collisions with walls or dense areas.

**Connecting Points**

After obtaining the pivot points, a Catmull-Rom spline is used to generate a smooth trajectory passing through them.

For each pair of consecutive pivots, the spline interpolates points using the two neighboring pivots on each side, producing a continuous path with no sharp turns. The segment between pivots is divided into a fixed number of points, ensuring uniform spacing along the trajectory. 

This results in a high-resolution path that the camera can follow smoothly during the indoor flythrough.

## 4. Object Detection

For indoor scenes, I applied YOLO-based object detection to better understand the content of the environment. Outdoor scenes are not processed with YOLO because the pre-trained model often misclassifies common outdoor structures (e.g., labeling everything as “surfboard”).

![Example: a detected bench in an indoor scene](imgs/detected_bench.png)

## 5. Camera Movement

Camera orientation is handled differently depending on the scene type:
- **Outdoor scenes:** The camera always looks toward the pivot point;
- **Indoor scenes:** The camera is oriented along the forward tangent of the path, essentially looking at the next pivot point in the trajectory.

# Future work

## Collision-Aware Path Planning

I explored methods to make indoor flythroughs collision-aware. The idea was to prevent the camera from intersecting walls, furniture, and other splats.

### Approach attempted:

**1. Voxel map construction:**

The scene was voxelized, with each voxel marked as occupied if the local point density exceeded a threshold. This produced a 3D occupancy grid highlighting areas the camera should avoid.

**2. Anchor-based DFS exploration:**

Starting from a central point (anchor) in the scene, a depth-first search (DFS) was used to explore free voxels and propose candidate camera positions along the path.

**3. Collision checking and path repair:**

After generating a preliminary trajectory, I checked each path point against the voxel map. Points inside occupied voxels were flagged, and attempts were made to re-route the path using breadth-first search (BFS) to navigate around obstacles.

**Outcome:**
Despite these efforts, the method did not reliably produce collision-free trajectories. Path adjustments often caused unnatural detours. While the idea remains promising, a more robust approach is required.

## Full Scene Rendering

Currently, the system renders only the Gaussian splat point cloud, which limits the scene realism. Future work could involve exporting the computed camera trajectories and applying them in a more advanced rendering engine. This would allow producing high-quality cinematic sequences while still leveraging the automatically generated paths from the system.

## Improved Path Planning via Learning

Current path-building logic is heuristic and does not account for scene complexity. Future work could explore using a reinforcement learning (RL) agent to optimize camera trajectories. The agent’s objective could be to maximize the distance traveled along interesting regions while simultaneously minimizing collisions with dense areas or obstacles. This approach would produce smoother, safer, and more visually engaging flythroughs that adapt intelligently to different scene geometries.
