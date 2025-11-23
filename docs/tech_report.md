# 3D Splats Cinematic Agent Technical Report

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

TODO

## 4. Object Detection

TODO

## 5. Camera Movement

TODO