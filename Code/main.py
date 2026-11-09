"""
CP467 Assignment 3: Object Identification and Localization with Local Features
Starter code. Fill in every function marked TODO.

Run from anywhere with:   python Code/main.py
All paths are relative to the submission folder, so the code runs on any computer.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths (do not change)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJECTS_DIR = os.path.join(ROOT, "Objects")
SCENES_DIR = os.path.join(ROOT, "Scenes")
DETECTED_DIR = os.path.join(ROOT, "Detected Objects")
TESTING_FILE = os.path.join(ROOT, "testing.txt")

OBJECT_IDS = ["O1", "O2", "O3", "O4"]
SCENE_NAMES = [f"S{s}_{v}.jpg" for s in (1, 2) for v in ("front", "left", "right")]

"""
TODO: Task 2: fill in the object names (same as your Object Mapping sheet).
They are drawn as labels on the bounding boxes.
"""
OBJECT_NAMES = {
    "O1": "Drink Carton",
    "O2": "Lotion Tube",
    "O3": "Candy bag",
    "O4": "Tape", 
}

# ---------------------------------------------------------------------------
# Parameters you will tune
# ---------------------------------------------------------------------------
RATIO_TEST = 0.60      # Lowe's ratio for keeping a match
MATCH_THRESHOLD = 26   # object is "present" if count >= this value


def load_image(path):
    """Load an image and stop with a clear message if it is missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_and_describe(image):
    """
    TODO: Detect keypoints and compute descriptors.
    Return (keypoints, descriptors).
    Hint: OpenCV has built-in SIFT and ORB detectors that return keypoints and
    descriptors in a single call.
    """
    return cv2.SIFT_create().detectAndCompute(image, None)


def match_descriptors(desc_object, desc_scene):
    """
    TODO: Match object descriptors to scene descriptors and keep good matches
    using the ratio test (RATIO_TEST). Return a list of cv2.DMatch.
    Hint: OpenCV has a brute-force matcher that can return the two nearest
    neighbours of each descriptor, which is what the ratio test needs.
    """
    if (desc_object is None or desc_scene is None
            or len(desc_object) == 0 or len(desc_scene) < 2):
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(desc_object, desc_scene, k=2)
    return [pair[0] for pair in neighbors
            if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]


def is_present(good_matches):
    """
    TODO: Decide whether the object is present in the scene.
    Return True or False. Record the counts you see in Results.xlsx (Match Counts).
    """
    return len(good_matches) >= MATCH_THRESHOLD


def locate_object(kp_scene, good_matches):
    """
    TODO (Task 2): Find a bounding box for the object in the scene from the
    scene positions of its good matches.
    1. Collect the scene points: kp_scene[m.trainIdx].pt for each good match.
    2. Remove stray points far from the object (for example, keep only points
       within a fixed distance of the median point).
    3. Return the bounding rectangle (x, y, w, h) of the remaining points, or None
       if too few points remain. OpenCV has a built-in function for this.
    """
    if len(good_matches) < 4:
        return None
    points = np.array([kp_scene[m.trainIdx].pt for m in good_matches],
                      dtype=np.float32)
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))
    radius = median_distance + 3.0 * 1.4826 * mad
    reliable = np.unique(points[distances <= radius], axis=0)
    if len(reliable) < 4:
        return None
    box = cv2.boundingRect(reliable)
    return box if box[2] > 1 and box[3] > 1 else None


def draw_box(scene_image, box, label=None):
    """Draw an axis-aligned bounding box and its label on the scene."""
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(scene_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
    if label:
        cv2.putText(scene_image, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return scene_image


def main():
    os.makedirs(DETECTED_DIR, exist_ok=True)

    # Features for every object image
    objects = {}
    for oid in OBJECT_IDS:
        img = load_image(os.path.join(OBJECTS_DIR, f"{oid}.jpg"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detect_and_describe(gray)
        objects[oid] = (kp, desc, gray.shape)

    lines = []
    for scene_name in SCENE_NAMES:
        scene = load_image(os.path.join(SCENES_DIR, scene_name))
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        kp_s, desc_s = detect_and_describe(scene_gray)

        detected = []
        output = scene.copy()
        for oid in OBJECT_IDS:
            kp_o, desc_o, shape_o = objects[oid]
            good = match_descriptors(desc_o, desc_s)
            print(f"{scene_name}  {oid}: {len(good)} good matches")
            if is_present(good):
                detected.append(oid)
                box = locate_object(kp_s, good)
                if box is not None:
                    draw_box(output, box, OBJECT_NAMES.get(oid) or None)

        # Task 2 output image (one per scene, even with no detections)
        out_name = scene_name.replace(".jpg", "_bb.jpg")
        cv2.imwrite(os.path.join(DETECTED_DIR, out_name), output)

        # Task 1 output line, in the required format
        lines.append(f"{scene_name}: {', '.join(detected) if detected else 'none'}")

    with open(TESTING_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TESTING_FILE}")


if __name__ == "__main__":
    main()
