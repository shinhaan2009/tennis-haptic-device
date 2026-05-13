#!/usr/bin/env python3
"""
Visually impaired tennis haptic feedback device.

Pipeline:
    tennis match video -> tennis ball detection/tracking -> court homography
    -> 6x10 top-down grid -> motor id 0..59 -> optional USB serial command.

Before processing starts, the script asks for:
    1. Four court-plane corners for the homography.
    2. A detection ROI polygon where yellow-ball candidates are allowed.

It also learns the calibrated court-view color distribution from the first frame.
When a broadcast cutaway/close-up no longer looks like that court view, detection
is paused and M-1 is sent.

Dependencies:
    Python 3.10+
    pip install opencv-python ultralytics numpy tqdm pyserial

Typical usage:
    Simulation only:
        python tennis_haptic_device.py --video tennis_match.mp4 --mode sim

    Hardware mode:
        python tennis_haptic_device.py --video tennis_match.mp4 --mode hardware --serial_port COM3
        python tennis_haptic_device.py --video tennis_match.mp4 --mode hardware --serial_port /dev/ttyUSB0

Serial protocol sent to ESP32 / Arduino Mega:
    M23\n     -> activate motor 23
    M-1\n     -> all motors off

TrackNet later:
    TrackNet-style tennis ball detectors usually output a heatmap over one or more
    consecutive frames. To switch from YOLO to TrackNet, keep the public
    detect_and_track_ball() function and replace detect_yolo_ball() with a
    function that returns DetectionCandidate(center=(x, y), confidence=...).
    The Kalman tracking, homography, grid mapping, serial, CSV, and visualization
    code can remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# Easy-to-change constants
# =============================================================================

# Input / output defaults
DEFAULT_VIDEO_PATH = "tennis_match.mp4"
DEFAULT_OUTPUT_VIDEO_PATH = "tennis_haptic_6x10.mp4"
DEFAULT_CSV_LOG_PATH = "ball_grid_log.csv"

# Top-down court size. Requirement: width=1000, height=600.
TOPDOWN_WIDTH = 1000
TOPDOWN_HEIGHT = 600
GRID_ROWS = 6
GRID_COLS = 10
CELL_WIDTH = TOPDOWN_WIDTH // GRID_COLS     # 100 px
CELL_HEIGHT = TOPDOWN_HEIGHT // GRID_ROWS   # 100 px

# User click order for homography.
COURT_POINT_LABELS = [
    "1 near left baseline corner",
    "2 near right baseline corner",
    "3 far right sideline corner",
    "4 far left sideline corner",
]

# Interactive point selection display limit.
MAX_SELECTION_WINDOW_WIDTH = 1400
MAX_SELECTION_WINDOW_HEIGHT = 900

# YOLO detector defaults.
# This Hugging Face repo provides a YOLOv8n tennis-ball fine-tuned weight file.
# You can replace this URL with your own Roboflow / Hugging Face / local model.
DEFAULT_YOLO_MODEL_URL = (
    "https://huggingface.co/RJTPP/tennis-ball-detection/resolve/main/best.pt"
)
DEFAULT_YOLO_MODEL_PATH = "models/tennis_ball_yolov8_best.pt"
GENERIC_YOLO_FALLBACK_MODEL = "yolov8n.pt"  # COCO model, class "sports ball".
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_YOLO_IMAGE_SIZE = 640

# YOLO bbox sanity filters. These reject obvious false positives.
YOLO_MIN_BOX_PIXELS = 3
YOLO_MAX_BOX_FRACTION = 0.18  # max box width/height as a fraction of frame max side.
YOLO_MIN_ASPECT_RATIO = 0.25
YOLO_MAX_ASPECT_RATIO = 4.00

# HSV fallback for fluorescent yellow/green tennis balls.
# Broadcast lighting varies a lot, so keep these easy to tune.
HSV_RANGES = [
    ((18, 45, 80), (48, 255, 255)),   # yellow tennis ball, includes dim/blurred pixels
    ((49, 35, 75), (86, 255, 255)),   # green-yellow tennis ball
]
HSV_MIN_CONTOUR_AREA = 2
HSV_MAX_CONTOUR_AREA = 1800
HSV_MIN_CIRCULARITY = 0.10
HSV_MIN_ASPECT_RATIO = 0.18
HSV_MAX_ASPECT_RATIO = 5.50
HSV_MORPH_KERNEL_SIZE = 3
YELLOW_SCORE_WEIGHT = 0.45

# Motion fallback for very small/blurred balls that YOLO misses.
# This is not used as a standalone truth source; candidates are scored against
# color, size, and the Kalman track so players/lines are less likely to win.
MOTION_DIFF_THRESHOLD = 18
MOTION_MIN_CONTOUR_AREA = 4
MOTION_MAX_CONTOUR_AREA = 1800
MOTION_MAX_BOX_SIDE = 80
MOTION_MIN_CIRCULARITY = 0.10
MOTION_MORPH_KERNEL_SIZE = 3
CANDIDATE_PROXIMITY_SCALE_PX = 180.0

# Restrict fallback detectors to the selected court polygon with a margin.
COURT_ROI_MARGIN_PX = 80
DEFAULT_ROI_MODE = "manual"

# Scene gate: if the broadcast cuts away from the calibrated court view, pause.
# These defaults are intentionally permissive. A false pause is worse than a
# short false positive because a pause prevents ball detection from running.
DEFAULT_COURT_MIN_COLOR_RATIO = 0.06
DEFAULT_COURT_MIN_HIST_SIMILARITY = 0.10
DEFAULT_SCENE_GATE_MODE = "soft"
DEFAULT_SCENE_GATE_CONSECUTIVE_FAILS = 8
COURT_COLOR_HUE_MARGIN = 12
COURT_COLOR_SAT_MARGIN = 45
COURT_COLOR_VAL_MARGIN = 60
COURT_SCENE_MIN_TRAIN_PIXELS = 500

# Kalman tracking.
DEFAULT_MAX_MISSING_FRAMES = 12
DEFAULT_MAX_TRACKING_JUMP_PX = 240.0
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 5e-2
KALMAN_ERROR_COVARIANCE = 1.0
PREDICTED_CONFIDENCE_DECAY = 0.80

# Processing / playback.
DEFAULT_FRAME_STRIDE = 1
DEFAULT_DISPLAY_SCALE = 0.75
DEFAULT_TOPDOWN_BACKGROUND = "synthetic"

# Serial / hardware.
DEFAULT_SERIAL_BAUD = 115200
DEFAULT_SERIAL_DEBOUNCE_MS = 75
SERIAL_BOOT_WAIT_SEC = 2.0

# Visualization colors, BGR order for OpenCV.
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 80, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class DetectionCandidate:
    """Raw detection before temporal smoothing."""

    center: tuple[float, float]
    confidence: float
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    source: str
    yellow_score: float = 0.0


@dataclass
class BallObservation:
    """Tracked ball position after Kalman smoothing/prediction."""

    center: Optional[tuple[float, float]]
    confidence: float
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    source: str
    predicted: bool


@dataclass
class DetectorBundle:
    """YOLO model wrapper. model=None means HSV-only fallback."""

    model: Any
    model_path: Optional[str]
    source_name: str


@dataclass
class GridState:
    row: int
    col: int
    motor_id: int
    inside: bool


@dataclass
class DebouncedMotorState:
    """Holds the last hardware motor to avoid rapid flicker."""

    current_motor_id: Optional[int] = None
    last_change_time: float = 0.0


@dataclass
class RoiSelection:
    """Detection ROI mask plus optional polygon points for visualization."""

    mask: Optional[np.ndarray]
    points: Optional[np.ndarray]
    source: str


@dataclass
class CourtSceneGate:
    """Reference court-view color model used to pause during broadcast cutaways."""

    roi_mask: np.ndarray
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    reference_hist: np.ndarray
    min_color_ratio: float
    min_hist_similarity: float
    mode: str
    consecutive_fail_limit: int
    consecutive_fail_count: int = 0

    def score(self, frame: np.ndarray) -> tuple[bool, float, float]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        color_mask = cv2.bitwise_and(color_mask, self.roi_mask)

        roi_area = max(int(cv2.countNonZero(self.roi_mask)), 1)
        color_ratio = cv2.countNonZero(color_mask) / float(roi_area)

        current_hist = compute_hs_histogram(hsv, self.roi_mask)
        hist_similarity = float(cv2.compareHist(self.reference_hist, current_hist, cv2.HISTCMP_INTERSECT))

        if self.mode == "off":
            return True, float(color_ratio), hist_similarity

        # In soft mode, either metric can keep the court-view alive. This avoids
        # blocking detection when lighting, camera exposure, or compression shifts.
        if self.mode == "soft":
            frame_passed = color_ratio >= self.min_color_ratio or hist_similarity >= self.min_hist_similarity
        else:
            frame_passed = color_ratio >= self.min_color_ratio and hist_similarity >= self.min_hist_similarity

        if frame_passed:
            self.consecutive_fail_count = 0
            visible = True
        else:
            self.consecutive_fail_count += 1
            visible = self.consecutive_fail_count < self.consecutive_fail_limit

        return visible, float(color_ratio), hist_similarity


# =============================================================================
# Utility helpers
# =============================================================================


def ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 2,
) -> None:
    """Readable text with black outline."""

    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_BLACK,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_height:
        return image
    scale = target_height / float(h)
    target_width = int(round(w * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch-like tensors or arrays to numpy without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def tensor_scalar(value: Any) -> float:
    arr = tensor_to_numpy(value).reshape(-1)
    return float(arr[0])


def get_class_name(names: Any, class_id: int) -> str:
    if names is None:
        return ""
    if isinstance(names, dict):
        return str(names.get(class_id, ""))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def class_count(names: Any) -> int:
    if names is None:
        return 0
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, (list, tuple)):
        return len(names)
    return 0


def is_ball_class(names: Any, class_id: int) -> bool:
    """
    Accept tennis-ball fine-tuned classes and COCO sports-ball.
    If the model has only one class, treat it as the ball class.
    """

    name = get_class_name(names, class_id).lower().replace("_", " ").replace("-", " ")
    if class_count(names) == 1:
        return True
    return ("ball" in name) or ("tennis" in name) or ("sports ball" in name)


def download_file(url: str, destination: str | Path) -> Optional[str]:
    """Download model weights using only the Python standard library."""

    dest = Path(destination)
    ensure_parent_dir(dest)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    print(f"Downloading tennis-ball YOLO model:\n  {url}\n  -> {dest}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            total = int(response.headers.get("Content-Length", "0") or "0")
            with open(tmp_dest, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc="Model download",
            ) as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp_dest.replace(dest)
        print(f"Model downloaded: {dest}")
        return str(dest)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"WARNING: Could not download YOLO tennis-ball model: {exc}")
        try:
            if tmp_dest.exists():
                tmp_dest.unlink()
        except OSError:
            pass
        return None


# =============================================================================
# Court perspective transform
# =============================================================================


def select_court_points(frame: np.ndarray) -> np.ndarray:
    """
    Open an interactive window and collect 4 court corner clicks.

    Click order:
        1. Near left baseline corner
        2. Near right baseline corner
        3. Far right sideline corner
        4. Far left sideline corner
    """

    original_h, original_w = frame.shape[:2]
    scale = min(
        1.0,
        MAX_SELECTION_WINDOW_WIDTH / float(original_w),
        MAX_SELECTION_WINDOW_HEIGHT / float(original_h),
    )
    display_w = int(round(original_w * scale))
    display_h = int(round(original_h * scale))
    display_base = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)

    points: list[tuple[float, float]] = []
    window_name = "Select 4 court corners - r reset, q quit"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
            print(f"Selected point {len(points)}: ({points[-1][0]:.1f}, {points[-1][1]:.1f})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nCourt calibration: click exactly 4 visible court corners in this order:")
    for label in COURT_POINT_LABELS:
        print(f"  {label}")
    print("Press r to reset points, q or Esc to quit calibration.\n")

    while True:
        canvas = display_base.copy()
        draw_text(canvas, "Click 4 court corners in order. r=reset, q=quit", (20, 35), 0.8)
        next_label = COURT_POINT_LABELS[len(points)] if len(points) < 4 else "done"
        draw_text(canvas, f"Next: {next_label}", (20, 70), 0.75, COLOR_YELLOW)

        scaled_points: list[tuple[int, int]] = []
        for idx, (px, py) in enumerate(points):
            sx, sy = int(round(px * scale)), int(round(py * scale))
            scaled_points.append((sx, sy))
            cv2.circle(canvas, (sx, sy), 7, COLOR_RED, -1, cv2.LINE_AA)
            draw_text(canvas, str(idx + 1), (sx + 10, sy - 10), 0.7, COLOR_RED)

        if len(scaled_points) >= 2:
            for a, b in zip(scaled_points[:-1], scaled_points[1:]):
                cv2.line(canvas, a, b, COLOR_ORANGE, 2, cv2.LINE_AA)
        if len(scaled_points) == 4:
            cv2.line(canvas, scaled_points[3], scaled_points[0], COLOR_ORANGE, 2, cv2.LINE_AA)
            draw_text(canvas, "4 points selected. Press Enter/Space to accept.", (20, 105), 0.75, COLOR_GREEN)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Court point selection cancelled by user.")
        if key == ord("r"):
            points.clear()
            print("Court points reset.")
        if len(points) == 4 and key in (13, 10, 32):
            break
        if len(points) == 4:
            # Accept automatically after a short display pause if no key is pressed.
            cv2.waitKey(400)
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


def build_homography(src_points: np.ndarray) -> np.ndarray:
    """Map clicked court quadrilateral to exact 1000x600 top-down rectangle."""

    dst_points = np.array(
        [
            [0, TOPDOWN_HEIGHT - 1],              # near left baseline -> bottom left
            [TOPDOWN_WIDTH - 1, TOPDOWN_HEIGHT - 1],  # near right baseline -> bottom right
            [TOPDOWN_WIDTH - 1, 0],               # far right sideline -> top right
            [0, 0],                               # far left sideline -> top left
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)


def apply_homography(
    frame: np.ndarray,
    homography: np.ndarray,
    ball_xy: Optional[tuple[float, float]],
) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
    """Warp frame and transform one ball center into top-down coordinates."""

    warped = cv2.warpPerspective(frame, homography, (TOPDOWN_WIDTH, TOPDOWN_HEIGHT))
    if ball_xy is None:
        return warped, None

    point = np.array([[[ball_xy[0], ball_xy[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)[0, 0]
    return warped, (float(transformed[0]), float(transformed[1]))


def make_court_roi_mask(
    frame_shape: tuple[int, ...],
    court_points: np.ndarray,
    margin_px: int = COURT_ROI_MARGIN_PX,
) -> np.ndarray:
    """
    Build a mask around the clicked court plane.

    Ball candidates outside this polygon are usually spectators, players,
    graphics, or court-line fragments. A margin keeps high balls and calibration
    imprecision from being clipped too aggressively.
    """

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = court_points.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if margin_px > 0:
        kernel_size = max(3, margin_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def select_detection_roi(frame: np.ndarray, fallback_roi: RoiSelection) -> RoiSelection:
    """
    Let the user draw the spatial range where yellow-ball detection is allowed.

    This is separate from the four court-corner clicks. In broadcast footage,
    yellow shirts, towels, ball kids, or ads may exist outside the useful play
    area. Masking them before detection is the highest-impact false-positive fix.
    """

    original_h, original_w = frame.shape[:2]
    scale = min(
        1.0,
        MAX_SELECTION_WINDOW_WIDTH / float(original_w),
        MAX_SELECTION_WINDOW_HEIGHT / float(original_h),
    )
    display_w = int(round(original_w * scale))
    display_h = int(round(original_h * scale))
    display_base = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)

    points: list[tuple[float, float]] = []
    window_name = "Select detection ROI - Enter accept, s use court ROI"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x / scale, y / scale))
            print(f"Detection ROI point {len(points)}: ({points[-1][0]:.1f}, {points[-1][1]:.1f})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nDetection ROI setup:")
    print("  Click a polygon enclosing only the region where the ball should be considered.")
    print("  Press Enter/Space to accept, r to reset, s to use the court ROI, q/Esc to cancel.\n")

    while True:
        canvas = display_base.copy()
        draw_text(canvas, "Click allowed ball-detection ROI. Enter=accept, s=court ROI", (20, 35), 0.72)
        draw_text(canvas, "Keep yellow clothes/ads outside this polygon when possible.", (20, 70), 0.68, COLOR_YELLOW)

        scaled_points: list[tuple[int, int]] = []
        for idx, (px, py) in enumerate(points):
            sx, sy = int(round(px * scale)), int(round(py * scale))
            scaled_points.append((sx, sy))
            cv2.circle(canvas, (sx, sy), 6, COLOR_BLUE, -1, cv2.LINE_AA)
            draw_text(canvas, str(idx + 1), (sx + 9, sy - 8), 0.55, COLOR_BLUE, 1)

        if len(scaled_points) >= 2:
            for a, b in zip(scaled_points[:-1], scaled_points[1:]):
                cv2.line(canvas, a, b, COLOR_BLUE, 2, cv2.LINE_AA)
        if len(scaled_points) >= 3:
            cv2.line(canvas, scaled_points[-1], scaled_points[0], COLOR_BLUE, 2, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Detection ROI selection cancelled by user.")
        if key == ord("r"):
            points.clear()
            print("Detection ROI points reset.")
        if key == ord("s"):
            cv2.destroyWindow(window_name)
            print("Using court polygon as detection ROI.")
            return fallback_roi
        if key in (13, 10, 32):
            if len(points) >= 3:
                break
            print("Need at least 3 points for a detection ROI polygon. Press s to use court ROI.")

    cv2.destroyWindow(window_name)

    pts = np.array(points, dtype=np.float32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], 255)
    print(f"Manual detection ROI selected with {len(points)} points.")
    return RoiSelection(mask=mask, points=pts, source="manual")


def compute_yellow_score(
    frame: np.ndarray,
    bbox_xyxy: Optional[tuple[int, int, int, int]],
) -> float:
    """Return how much of a candidate patch matches tennis-ball yellow/green."""

    if bbox_xyxy is None:
        return 0.0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad = 3
    x1 = int(np.clip(x1 - pad, 0, w - 1))
    y1 = int(np.clip(y1 - pad, 0, h - 1))
    x2 = int(np.clip(x2 + pad, x1 + 1, w))
    y2 = int(np.clip(y2 + pad, y1 + 1, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yellow_pixels = 0
    for lower, upper in HSV_RANGES:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        yellow_pixels += int(cv2.countNonZero(cv2.inRange(hsv_patch, lower_np, upper_np)))
    return float(np.clip(yellow_pixels / float(max(patch.shape[0] * patch.shape[1], 1)), 0.0, 1.0))


def compute_hs_histogram(hsv: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([hsv], [0, 1], roi_mask, [32, 16], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 1e-6:
        hist = np.zeros((32, 16), dtype=np.float32)
        hist[0, 0] = 1.0
        return hist
    return (hist / total).astype(np.float32)


def train_court_scene_gate(
    first_frame: np.ndarray,
    court_mask: np.ndarray,
    min_color_ratio: float,
    min_hist_similarity: float,
    mode: str,
    consecutive_fail_limit: int,
) -> Optional[CourtSceneGate]:
    """
    Learn a simple court-view model from the calibrated first frame.

    During player close-ups or cutaways, the fixed court ROI no longer contains
    the same court-color distribution, so detection and serial output are paused.
    """

    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    roi_pixels = hsv[court_mask > 0]
    if len(roi_pixels) < COURT_SCENE_MIN_TRAIN_PIXELS:
        print("WARNING: Not enough pixels to train court scene gate; disabling scene pause.")
        return None

    # Ignore nearly white lines/overlays and very dark pixels when learning the
    # dominant court surface, but use the full ROI for the histogram.
    train_pixels = roi_pixels[(roi_pixels[:, 1] > 35) & (roi_pixels[:, 2] > 45)]
    if len(train_pixels) < COURT_SCENE_MIN_TRAIN_PIXELS:
        train_pixels = roi_pixels

    h_low, s_low, v_low = np.percentile(train_pixels, 12, axis=0)
    h_high, s_high, v_high = np.percentile(train_pixels, 88, axis=0)

    hsv_lower = np.array(
        [
            max(0, int(h_low) - COURT_COLOR_HUE_MARGIN),
            max(0, int(s_low) - COURT_COLOR_SAT_MARGIN),
            max(0, int(v_low) - COURT_COLOR_VAL_MARGIN),
        ],
        dtype=np.uint8,
    )
    hsv_upper = np.array(
        [
            min(179, int(h_high) + COURT_COLOR_HUE_MARGIN),
            min(255, int(s_high) + COURT_COLOR_SAT_MARGIN),
            min(255, int(v_high) + COURT_COLOR_VAL_MARGIN),
        ],
        dtype=np.uint8,
    )

    # If hue is too broad because of logos/lines/shadows, rely more on histogram.
    if hsv_upper[0] <= hsv_lower[0] or (int(hsv_upper[0]) - int(hsv_lower[0])) > 80:
        hsv_lower[0] = 0
        hsv_upper[0] = 179

    reference_hist = compute_hs_histogram(hsv, court_mask)
    gate = CourtSceneGate(
        roi_mask=court_mask,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        reference_hist=reference_hist,
        min_color_ratio=min_color_ratio,
        min_hist_similarity=min_hist_similarity,
        mode=mode,
        consecutive_fail_limit=consecutive_fail_limit,
    )

    visible, color_ratio, hist_similarity = gate.score(first_frame)
    print(
        "Court scene gate trained: "
        f"initial color_ratio={color_ratio:.2f}, hist_similarity={hist_similarity:.2f}, "
        f"visible={visible}"
    )
    return gate


# =============================================================================
# Ball detection and tracking
# =============================================================================


def load_ball_detector(
    model_path: Optional[str],
    model_url: str,
    allow_download: bool,
) -> DetectorBundle:
    """
    Load a tennis-ball YOLO detector.

    Priority:
        1. Explicit --model_path
        2. Previously downloaded DEFAULT_YOLO_MODEL_PATH
        3. Auto-download DEFAULT_YOLO_MODEL_URL
        4. Generic Ultralytics yolov8n.pt COCO model
        5. HSV-only fallback if YOLO cannot load
    """

    try:
        from ultralytics import YOLO
    except ImportError:
        print("WARNING: ultralytics is not installed. Falling back to HSV-only detection.")
        return DetectorBundle(model=None, model_path=None, source_name="HSV-only")

    resolved_model: Optional[str] = None

    if model_path:
        model_path_obj = Path(model_path).expanduser()
        if model_path_obj.exists():
            resolved_model = str(model_path_obj)
        else:
            print(f"WARNING: --model_path does not exist: {model_path}")

    default_model_path = Path(DEFAULT_YOLO_MODEL_PATH)
    if resolved_model is None and default_model_path.exists():
        resolved_model = str(default_model_path)

    if resolved_model is None and allow_download and model_url:
        resolved_model = download_file(model_url, DEFAULT_YOLO_MODEL_PATH)

    if resolved_model is None:
        resolved_model = GENERIC_YOLO_FALLBACK_MODEL
        print(
            "WARNING: Using generic yolov8n.pt fallback. "
            "For best accuracy, provide a tennis-ball fine-tuned model with --model_path."
        )

    try:
        model = YOLO(resolved_model)
        print(f"Loaded YOLO detector: {resolved_model}")
        return DetectorBundle(model=model, model_path=resolved_model, source_name="YOLO")
    except Exception as exc:
        print(f"WARNING: Could not load YOLO model '{resolved_model}': {exc}")
        print("Falling back to HSV-only detection.")
        return DetectorBundle(model=None, model_path=None, source_name="HSV-only")


def detect_yolo_candidates(
    frame: np.ndarray,
    detector: DetectorBundle,
    conf_threshold: float,
    imgsz: int,
    device: Optional[str],
    court_mask: Optional[np.ndarray] = None,
) -> list[DetectionCandidate]:
    """Run YOLO and return all ball-like candidates passing basic filters."""

    if detector.model is None:
        return []

    h, w = frame.shape[:2]
    max_box_side = max(w, h) * YOLO_MAX_BOX_FRACTION

    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "conf": conf_threshold,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    try:
        results = detector.model.predict(**predict_kwargs)
    except Exception as exc:
        print(f"WARNING: YOLO prediction failed on this frame: {exc}")
        return []

    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", None) or getattr(detector.model, "names", None)
    candidates: list[DetectionCandidate] = []

    for box in boxes:
        try:
            xyxy = tensor_to_numpy(box.xyxy[0]).astype(float)
            confidence = float(tensor_scalar(box.conf[0]))
            class_id = int(tensor_scalar(box.cls[0])) if getattr(box, "cls", None) is not None else -1
        except Exception:
            continue

        if confidence < conf_threshold:
            continue
        if not is_ball_class(names, class_id):
            continue

        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        if bw < YOLO_MIN_BOX_PIXELS or bh < YOLO_MIN_BOX_PIXELS:
            continue
        if bw > max_box_side or bh > max_box_side:
            continue
        aspect = bw / max(bh, 1.0)
        if not (YOLO_MIN_ASPECT_RATIO <= aspect <= YOLO_MAX_ASPECT_RATIO):
            continue

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        if court_mask is not None:
            ix = int(np.clip(round(cx), 0, w - 1))
            iy = int(np.clip(round(cy), 0, h - 1))
            if court_mask[iy, ix] == 0:
                continue

        bbox = (int(x1), int(y1), int(x2), int(y2))
        candidates.append(
            DetectionCandidate(
                center=(cx, cy),
                confidence=confidence,
                bbox_xyxy=bbox,
                source="YOLO",
                yellow_score=compute_yellow_score(frame, bbox),
            )
        )

    return candidates


def detect_yolo_ball(
    frame: np.ndarray,
    detector: DetectorBundle,
    conf_threshold: float,
    imgsz: int,
    device: Optional[str],
    court_mask: Optional[np.ndarray] = None,
) -> Optional[DetectionCandidate]:
    """Compatibility wrapper: return the highest-confidence YOLO candidate."""

    candidates = detect_yolo_candidates(frame, detector, conf_threshold, imgsz, device, court_mask)
    return max(candidates, key=lambda item: item.confidence, default=None)


def detect_hsv_candidates(
    frame: np.ndarray,
    conf_threshold: float,
    court_mask: Optional[np.ndarray] = None,
) -> list[DetectionCandidate]:
    """
    HSV color fallback candidates for yellow/green tennis balls.

    This is intentionally conservative: it filters by area, aspect ratio, and
    circularity to avoid sending motors from court paint or line artifacts.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

    for lower, upper in HSV_RANGES:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lower_np, upper_np))

    if court_mask is not None:
        mask_total = cv2.bitwise_and(mask_total, court_mask)

    kernel = np.ones((HSV_MORPH_KERNEL_SIZE, HSV_MORPH_KERNEL_SIZE), dtype=np.uint8)
    # Close small gaps so a blurred tennis ball becomes one blob. Avoid opening:
    # in broadcast footage the ball may be only a few pixels, and erosion can
    # delete it entirely.
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DetectionCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < HSV_MIN_CONTOUR_AREA or area > HSV_MAX_CONTOUR_AREA:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < HSV_MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(float(h), 1.0)
        if not (HSV_MIN_ASPECT_RATIO <= aspect <= HSV_MAX_ASPECT_RATIO):
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            cx = x + w * 0.5
            cy = y + h * 0.5
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

        # Pseudo-confidence from yellow color + rough shape. The ROI already
        # removes most yellow clothes/ads, so keep tiny blurred balls alive.
        area_score = min(area / 45.0, 1.0)
        confidence = float(np.clip(0.45 + 0.35 * min(circularity, 1.0) + 0.20 * area_score, 0.0, 0.96))
        if confidence < conf_threshold:
            continue

        bbox = (x, y, x + w, y + h)
        candidates.append(
            DetectionCandidate(
                center=(float(cx), float(cy)),
                confidence=confidence,
                bbox_xyxy=bbox,
                source="HSV",
                yellow_score=compute_yellow_score(frame, bbox),
            )
        )

    return candidates


def detect_hsv_ball(
    frame: np.ndarray,
    conf_threshold: float,
    court_mask: Optional[np.ndarray] = None,
) -> Optional[DetectionCandidate]:
    """Compatibility wrapper: return the highest-confidence HSV candidate."""

    candidates = detect_hsv_candidates(frame, conf_threshold, court_mask)
    return max(candidates, key=lambda item: item.confidence, default=None)


class MotionBallDetector:
    """
    Small moving-object detector for balls missed by YOLO.

    It intentionally returns candidates, not final detections. Final selection is
    done with source confidence plus distance to the Kalman track.
    """

    def __init__(self) -> None:
        self.previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_gray = None

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        court_mask: Optional[np.ndarray] = None,
    ) -> list[DetectionCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        diff = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray

        _, mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        if court_mask is not None:
            mask = cv2.bitwise_and(mask, court_mask)

        kernel = np.ones((MOTION_MORPH_KERNEL_SIZE, MOTION_MORPH_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[DetectionCandidate] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MOTION_MIN_CONTOUR_AREA or area > MOTION_MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w > MOTION_MAX_BOX_SIDE or h > MOTION_MAX_BOX_SIDE:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0
            if perimeter > 0:
                circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < MOTION_MIN_CIRCULARITY:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                cx = x + w * 0.5
                cy = y + h * 0.5
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

            patch = hsv[y : y + h, x : x + w]
            if patch.size == 0:
                continue

            yellow_pixels = 0
            for lower, upper in HSV_RANGES:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                yellow_pixels += int(cv2.countNonZero(cv2.inRange(patch, lower_np, upper_np)))
            yellow_ratio = yellow_pixels / float(max(w * h, 1))

            # Motion alone is weak, but motion + tennis-ball color is useful.
            area_score = min(area / 120.0, 1.0)
            confidence = float(
                np.clip(
                    0.34 + 0.22 * min(circularity, 1.0) + 0.18 * area_score + 0.26 * min(yellow_ratio * 3.0, 1.0),
                    0.0,
                    0.88,
                )
            )
            if confidence < conf_threshold:
                continue

            source = "Motion+HSV" if yellow_ratio > 0.08 else "Motion"
            bbox = (x, y, x + w, y + h)
            candidates.append(
                DetectionCandidate(
                    center=(float(cx), float(cy)),
                    confidence=confidence,
                    bbox_xyxy=bbox,
                    source=source,
                    yellow_score=max(float(yellow_ratio), compute_yellow_score(frame, bbox)),
                )
            )

        return candidates


def candidate_score(
    candidate: DetectionCandidate,
    reference_center: Optional[tuple[float, float]],
) -> float:
    """Score candidates from mixed detectors using confidence and track continuity."""

    source_weight = {
        "YOLO": 0.90,
        "HSV": 1.00,
        "Motion+HSV": 0.94,
        "Motion": 0.58,
    }.get(candidate.source, 0.65)

    score = candidate.confidence * source_weight
    score += YELLOW_SCORE_WEIGHT * candidate.yellow_score
    if reference_center is not None:
        dx = candidate.center[0] - reference_center[0]
        dy = candidate.center[1] - reference_center[1]
        distance = math.hypot(dx, dy)
        proximity = math.exp(-distance / CANDIDATE_PROXIMITY_SCALE_PX)
        score += 0.35 * proximity
    else:
        score += 0.08

    return score


def choose_best_candidate(
    candidates: list[DetectionCandidate],
    reference_center: Optional[tuple[float, float]],
) -> Optional[DetectionCandidate]:
    if not candidates:
        return None
    return max(candidates, key=lambda item: candidate_score(item, reference_center))


class BallKalmanTracker:
    """Constant-velocity 2D Kalman tracker for smoothing and brief occlusions."""

    def __init__(self, max_missing_frames: int, max_jump_px: float) -> None:
        self.max_missing_frames = max_missing_frames
        self.max_jump_px = max_jump_px
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * KALMAN_ERROR_COVARIANCE
        self.initialized = False
        self.missing_frames = 0
        self.last_confidence = 0.0
        self.last_bbox: Optional[tuple[int, int, int, int]] = None
        self.last_source = "none"

    def reset(self) -> None:
        self.initialized = False
        self.missing_frames = 0
        self.last_confidence = 0.0
        self.last_bbox = None
        self.last_source = "none"
        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def reference_center(self) -> Optional[tuple[float, float]]:
        """Last corrected track position, used only for scoring new candidates."""

        if not self.initialized:
            return None
        return (float(self.kalman.statePost[0, 0]), float(self.kalman.statePost[1, 0]))

    def initialize(self, center: tuple[float, float]) -> None:
        x, y = center
        state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.kalman.statePre = state.copy()
        self.kalman.statePost = state.copy()
        self.initialized = True
        self.missing_frames = 0

    def update(self, candidate: Optional[DetectionCandidate]) -> Optional[BallObservation]:
        predicted_xy: Optional[tuple[float, float]] = None

        if self.initialized:
            prediction = self.kalman.predict()
            predicted_xy = (float(prediction[0, 0]), float(prediction[1, 0]))

        # Reject large low-confidence jumps; they are usually false positives.
        if candidate is not None and predicted_xy is not None:
            dx = candidate.center[0] - predicted_xy[0]
            dy = candidate.center[1] - predicted_xy[1]
            jump = math.hypot(dx, dy)
            if jump > self.max_jump_px and candidate.confidence < 0.70:
                candidate = None

        if candidate is not None:
            if not self.initialized:
                self.initialize(candidate.center)

            measurement = np.array([[candidate.center[0]], [candidate.center[1]]], dtype=np.float32)
            corrected = self.kalman.correct(measurement)
            center = (float(corrected[0, 0]), float(corrected[1, 0]))
            self.missing_frames = 0
            self.last_confidence = candidate.confidence
            self.last_bbox = candidate.bbox_xyxy
            self.last_source = candidate.source
            return BallObservation(
                center=center,
                confidence=candidate.confidence,
                bbox_xyxy=candidate.bbox_xyxy,
                source=candidate.source,
                predicted=False,
            )

        if self.initialized and predicted_xy is not None and self.missing_frames < self.max_missing_frames:
            self.missing_frames += 1
            confidence = self.last_confidence * (PREDICTED_CONFIDENCE_DECAY ** self.missing_frames)
            return BallObservation(
                center=predicted_xy,
                confidence=float(confidence),
                bbox_xyxy=self.last_bbox,
                source=f"Kalman({self.last_source})",
                predicted=True,
            )

        return None


def detect_and_track_ball(
    frame: np.ndarray,
    detector: DetectorBundle,
    tracker: BallKalmanTracker,
    motion_detector: MotionBallDetector,
    conf_threshold: float,
    imgsz: int,
    device: Optional[str],
    use_hsv_fallback: bool,
    use_motion_fallback: bool,
    hsv_only: bool,
    court_mask: Optional[np.ndarray],
) -> Optional[BallObservation]:
    """
    Primary YOLO detection with HSV/motion fallbacks, then Kalman tracking.

    This function is the main integration point if replacing YOLO with TrackNet.
    """

    candidates: list[DetectionCandidate] = []

    if not hsv_only:
        candidates.extend(detect_yolo_candidates(frame, detector, conf_threshold, imgsz, device, court_mask))

    if use_hsv_fallback:
        candidates.extend(detect_hsv_candidates(frame, conf_threshold, court_mask))

    if use_motion_fallback:
        candidates.extend(motion_detector.detect(frame, conf_threshold, court_mask))

    candidate = choose_best_candidate(candidates, tracker.reference_center())
    return tracker.update(candidate)


# =============================================================================
# Grid mapping and serial output
# =============================================================================


def map_to_motor(
    topdown_xy: Optional[tuple[float, float]],
    last_valid_state: Optional[GridState],
    outside_behavior: str,
) -> GridState:
    """
    Convert top-down ball coordinate to a 6x10 grid and motor id.

    outside_behavior:
        "off"  -> outside/no ball maps to motor -1
        "keep" -> outside/no ball keeps the last valid cell if available
    """

    if topdown_xy is None:
        if outside_behavior == "keep" and last_valid_state is not None:
            return last_valid_state
        return GridState(row=-1, col=-1, motor_id=-1, inside=False)

    x, y = topdown_xy
    inside = 0 <= x < TOPDOWN_WIDTH and 0 <= y < TOPDOWN_HEIGHT
    if not inside:
        if outside_behavior == "keep" and last_valid_state is not None:
            return last_valid_state
        return GridState(row=-1, col=-1, motor_id=-1, inside=False)

    grid_row = int(y // CELL_HEIGHT)
    grid_col = int(x // CELL_WIDTH)
    grid_row = int(np.clip(grid_row, 0, GRID_ROWS - 1))
    grid_col = int(np.clip(grid_col, 0, GRID_COLS - 1))
    motor_id = grid_row * GRID_COLS + grid_col
    return GridState(row=grid_row, col=grid_col, motor_id=motor_id, inside=True)


def debounce_motor_id(raw_motor_id: int, state: DebouncedMotorState, debounce_ms: int) -> int:
    """
    Accept the first motor immediately, then prevent faster-than-debounce changes.

    The main loop still calls serial write every processed frame. During the
    debounce window, the command remains the previous motor id to prevent flicker.
    """

    now = time.perf_counter()
    if state.current_motor_id is None:
        state.current_motor_id = raw_motor_id
        state.last_change_time = now
        return raw_motor_id

    if raw_motor_id != state.current_motor_id:
        elapsed_ms = (now - state.last_change_time) * 1000.0
        if debounce_ms <= 0 or elapsed_ms >= debounce_ms:
            state.current_motor_id = raw_motor_id
            state.last_change_time = now

    return int(state.current_motor_id)


def open_serial_connection(serial_port: str, baudrate: int) -> Any:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is not installed. Run: pip install pyserial") from exc

    try:
        ser = serial.Serial(
            port=serial_port,
            baudrate=baudrate,
            timeout=0,
            write_timeout=0,
        )
        print(f"Opened serial port {serial_port} at {baudrate} baud.")
        print(f"Waiting {SERIAL_BOOT_WAIT_SEC:.1f}s for board reset...")
        time.sleep(SERIAL_BOOT_WAIT_SEC)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        return ser
    except Exception as exc:
        raise RuntimeError(f"Could not open serial port {serial_port}: {exc}") from exc


def send_to_hardware(serial_conn: Any, motor_id: int) -> bool:
    """
    Send short command to ESP32 / Arduino Mega.

    Required format:
        M{motor_id}\n
    """

    if serial_conn is None:
        return False

    command = f"M{motor_id}\n".encode("ascii")
    try:
        serial_conn.write(command)
        return True
    except Exception as exc:
        print(f"WARNING: Serial write failed: {exc}")
        return False


# =============================================================================
# Visualization
# =============================================================================


def draw_original_overlay(
    frame: np.ndarray,
    observation: Optional[BallObservation],
    court_points: np.ndarray,
    detection_roi_points: Optional[np.ndarray] = None,
    scene_visible: bool = True,
    scene_scores: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    canvas = frame.copy()

    # Draw clicked court quadrilateral.
    pts = court_points.astype(int).reshape(-1, 2)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 7, COLOR_RED, -1, cv2.LINE_AA)
        draw_text(canvas, str(idx + 1), (int(x) + 10, int(y) - 10), 0.7, COLOR_RED)
    if len(pts) == 4:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=COLOR_ORANGE, thickness=2)

    if detection_roi_points is not None and len(detection_roi_points) >= 3:
        roi_pts = detection_roi_points.astype(int).reshape(-1, 1, 2)
        cv2.polylines(canvas, [roi_pts], isClosed=True, color=COLOR_BLUE, thickness=2)
        draw_text(canvas, "detection ROI", (int(roi_pts[0, 0, 0]) + 8, int(roi_pts[0, 0, 1]) + 18), 0.5, COLOR_BLUE, 1)

    if not scene_visible:
        draw_text(canvas, "COURT VIEW NOT VISIBLE - detection paused", (20, 110), 0.8, COLOR_RED)
    if scene_scores is not None:
        color_ratio, hist_similarity = scene_scores
        draw_text(canvas, f"court_gate color={color_ratio:.2f} hist={hist_similarity:.2f}", (20, 142), 0.55, COLOR_WHITE, 1)

    # Draw tracked ball.
    if observation is not None and observation.center is not None:
        cx, cy = int(round(observation.center[0])), int(round(observation.center[1]))
        color = COLOR_BLUE if observation.predicted else COLOR_YELLOW
        cv2.circle(canvas, (cx, cy), 10, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, color, -1, cv2.LINE_AA)
        if observation.bbox_xyxy is not None and not observation.predicted:
            x1, y1, x2, y2 = observation.bbox_xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{observation.source} conf={observation.confidence:.2f}"
        if observation.predicted:
            label += " predicted"
        draw_text(canvas, label, (max(cx + 12, 10), max(cy - 12, 25)), 0.55, color)

    return canvas


def make_synthetic_topdown_court() -> np.ndarray:
    """
    Draw a clean schematic court instead of displaying warped video.

    A homography is valid only for the flat court plane. The net is vertical, so
    it will always look badly stretched in a top-down warp. For haptic debugging,
    a synthetic court is clearer and physically more honest.
    """

    court = np.full((TOPDOWN_HEIGHT, TOPDOWN_WIDTH, 3), (38, 125, 78), dtype=np.uint8)

    # Subtle inner court surface.
    cv2.rectangle(court, (0, 0), (TOPDOWN_WIDTH - 1, TOPDOWN_HEIGHT - 1), (44, 148, 91), -1)

    # Approximate tennis reference lines on the normalized full court.
    net_y = TOPDOWN_HEIGHT // 2
    service_offset = int(round((21.0 / 78.0) * TOPDOWN_HEIGHT))
    upper_service_y = net_y - service_offset
    lower_service_y = net_y + service_offset

    line_color = (235, 235, 235)
    cv2.rectangle(court, (0, 0), (TOPDOWN_WIDTH - 1, TOPDOWN_HEIGHT - 1), line_color, 3)
    cv2.line(court, (0, net_y), (TOPDOWN_WIDTH, net_y), (25, 25, 25), 4, cv2.LINE_AA)
    cv2.line(court, (0, net_y), (TOPDOWN_WIDTH, net_y), line_color, 1, cv2.LINE_AA)
    cv2.line(court, (0, upper_service_y), (TOPDOWN_WIDTH, upper_service_y), line_color, 2, cv2.LINE_AA)
    cv2.line(court, (0, lower_service_y), (TOPDOWN_WIDTH, lower_service_y), line_color, 2, cv2.LINE_AA)
    cv2.line(court, (TOPDOWN_WIDTH // 2, upper_service_y), (TOPDOWN_WIDTH // 2, lower_service_y), line_color, 2)

    return court


def draw_topdown_overlay(
    warped_frame: np.ndarray,
    topdown_xy: Optional[tuple[float, float]],
    grid_state: GridState,
    background_mode: str = DEFAULT_TOPDOWN_BACKGROUND,
) -> np.ndarray:
    if background_mode == "warp":
        canvas = warped_frame.copy()
    else:
        canvas = make_synthetic_topdown_court()

    # Highlight current active cell in green.
    if 0 <= grid_state.row < GRID_ROWS and 0 <= grid_state.col < GRID_COLS:
        x1 = grid_state.col * CELL_WIDTH
        y1 = grid_state.row * CELL_HEIGHT
        x2 = x1 + CELL_WIDTH
        y2 = y1 + CELL_HEIGHT
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GREEN, -1)
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_GREEN, 3)

    # Grid lines. Each cell is exactly 100x100 px.
    for col in range(GRID_COLS + 1):
        x = col * CELL_WIDTH
        cv2.line(canvas, (x, 0), (x, TOPDOWN_HEIGHT), COLOR_WHITE, 1, cv2.LINE_AA)
    for row in range(GRID_ROWS + 1):
        y = row * CELL_HEIGHT
        cv2.line(canvas, (0, y), (TOPDOWN_WIDTH, y), COLOR_WHITE, 1, cv2.LINE_AA)

    # Optional cell ids, useful during wiring/debugging.
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            motor_id = row * GRID_COLS + col
            draw_text(
                canvas,
                str(motor_id),
                (col * CELL_WIDTH + 6, row * CELL_HEIGHT + 22),
                scale=0.45,
                color=COLOR_WHITE,
                thickness=1,
            )

    # Ball in top-down court.
    if topdown_xy is not None:
        x, y = int(round(topdown_xy[0])), int(round(topdown_xy[1]))
        if -50 <= x <= TOPDOWN_WIDTH + 50 and -50 <= y <= TOPDOWN_HEIGHT + 50:
            cv2.circle(canvas, (x, y), 11, COLOR_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 13, COLOR_BLACK, 2, cv2.LINE_AA)

    draw_text(canvas, "Top-down 6x10 haptic grid", (18, TOPDOWN_HEIGHT - 18), 0.65, COLOR_WHITE)
    return canvas


def make_side_by_side(
    original_vis: np.ndarray,
    topdown_vis: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    motor_id: int,
    fps_estimate: float,
) -> np.ndarray:
    left = resize_to_height(original_vis, TOPDOWN_HEIGHT)
    if topdown_vis.shape[:2] != (TOPDOWN_HEIGHT, TOPDOWN_WIDTH):
        right = cv2.resize(topdown_vis, (TOPDOWN_WIDTH, TOPDOWN_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        right = topdown_vis

    combined = np.hstack([left, right])
    text = f"frame={frame_index}  time={timestamp_sec:.2f}s  motor_id={motor_id}  FPS={fps_estimate:.1f}"
    draw_text(combined, text, (18, 30), 0.75, COLOR_YELLOW)
    return combined


# =============================================================================
# Main application
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time tennis ball to 6x10 haptic motor grid mapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input tennis match video path.")
    parser.add_argument("--mode", choices=["sim", "hardware"], default="sim", help="Simulation or hardware mode.")
    parser.add_argument("--serial_port", default=None, help="Serial port, e.g. COM3 or /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_SERIAL_BAUD, help="Serial baudrate.")
    parser.add_argument("--debounce_ms", type=int, default=DEFAULT_SERIAL_DEBOUNCE_MS, help="Motor change debounce.")
    parser.add_argument(
        "--outside_behavior",
        choices=["off", "keep"],
        default="off",
        help="What to send when ball is outside court or not tracked.",
    )

    parser.add_argument("--model_path", default=None, help="Path to custom tennis-ball YOLO .pt model.")
    parser.add_argument("--model_url", default=DEFAULT_YOLO_MODEL_URL, help="Auto-download model URL.")
    parser.add_argument("--no_model_download", action="store_true", help="Disable automatic model download.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_YOLO_IMAGE_SIZE, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--hsv_only", action="store_true", help="Skip YOLO and use non-neural HSV/motion fallback only.")
    parser.add_argument("--disable_hsv_fallback", action="store_true", help="Disable HSV fallback after YOLO misses.")
    parser.add_argument("--disable_motion_fallback", action="store_true", help="Disable frame-difference motion fallback.")
    parser.add_argument(
        "--roi_mode",
        choices=["manual", "court", "none"],
        default=DEFAULT_ROI_MODE,
        help="Detection ROI source. manual lets you draw an allowed yellow-ball region before processing.",
    )
    parser.add_argument(
        "--disable_court_roi",
        action="store_true",
        help="Deprecated alias for --roi_mode none.",
    )
    parser.add_argument("--disable_court_scene_gate", action="store_true", help="Do not pause detection on broadcast cutaways.")
    parser.add_argument(
        "--scene_gate_mode",
        choices=["soft", "strict", "off"],
        default=DEFAULT_SCENE_GATE_MODE,
        help="soft pauses only after repeated strong cutaway evidence; strict requires both court metrics every frame.",
    )
    parser.add_argument(
        "--scene_gate_consecutive_fails",
        type=int,
        default=DEFAULT_SCENE_GATE_CONSECUTIVE_FAILS,
        help="How many consecutive failed scene-gate frames are needed before pausing.",
    )
    parser.add_argument("--court_min_color_ratio", type=float, default=DEFAULT_COURT_MIN_COLOR_RATIO, help="Court-view pause gate color threshold.")
    parser.add_argument("--court_min_hist_similarity", type=float, default=DEFAULT_COURT_MIN_HIST_SIMILARITY, help="Court-view pause gate histogram threshold.")
    parser.add_argument("--max_missing", type=int, default=DEFAULT_MAX_MISSING_FRAMES, help="Kalman prediction frames.")
    parser.add_argument("--max_jump_px", type=float, default=DEFAULT_MAX_TRACKING_JUMP_PX, help="Tracking jump gate.")

    parser.add_argument("--frame_stride", type=int, default=DEFAULT_FRAME_STRIDE, help="Process every Nth frame.")
    parser.add_argument("--output_video", default=DEFAULT_OUTPUT_VIDEO_PATH, help="Visualization MP4 output path.")
    parser.add_argument("--csv_log", default=DEFAULT_CSV_LOG_PATH, help="CSV log output path.")
    parser.add_argument(
        "--topdown_background",
        choices=["synthetic", "warp"],
        default=DEFAULT_TOPDOWN_BACKGROUND,
        help="Right-panel background. synthetic avoids vertical-net homography distortion.",
    )
    parser.add_argument("--display_scale", type=float, default=DEFAULT_DISPLAY_SCALE, help="Display window scale.")
    parser.add_argument("--no_display", action="store_true", help="Do not show real-time visualization window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.frame_stride < 1:
        print("ERROR: --frame_stride must be >= 1")
        return 2
    if args.conf < 0.40:
        print("WARNING: Requirement asks for confidence threshold >= 0.4; clamping to 0.40.")
        args.conf = 0.40
    if args.mode == "hardware" and not args.serial_port:
        print("ERROR: hardware mode requires --serial_port, e.g. --serial_port COM3")
        return 2
    if args.disable_court_roi:
        args.roi_mode = "none"
    if args.disable_court_scene_gate:
        args.scene_gate_mode = "off"

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}")
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        return 2

    serial_conn = None
    video_writer: Optional[cv2.VideoWriter] = None
    csv_file = None
    pbar: Optional[tqdm] = None

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 1e-3 or math.isnan(source_fps):
            source_fps = 30.0
            print("WARNING: video FPS unavailable; using 30 FPS for timestamps/output.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            print("ERROR: could not read first frame.")
            return 2

        court_points = select_court_points(first_frame)
        homography = build_homography(court_points)
        court_mask_for_detection = make_court_roi_mask(first_frame.shape, court_points)
        court_mask_for_scene = make_court_roi_mask(first_frame.shape, court_points, margin_px=0)

        if args.roi_mode == "manual":
            detection_roi = select_detection_roi(
                first_frame,
                RoiSelection(mask=court_mask_for_detection, points=court_points, source="court"),
            )
        elif args.roi_mode == "court":
            detection_roi = RoiSelection(mask=court_mask_for_detection, points=court_points, source="court")
            print("Using clicked court polygon as detection ROI.")
        else:
            detection_roi = RoiSelection(mask=None, points=None, source="none")
            print("Detection ROI disabled; yellow candidates may include off-court objects.")

        scene_gate: Optional[CourtSceneGate] = None
        if args.scene_gate_mode != "off":
            scene_gate = train_court_scene_gate(
                first_frame=first_frame,
                court_mask=court_mask_for_scene,
                min_color_ratio=args.court_min_color_ratio,
                min_hist_similarity=args.court_min_hist_similarity,
                mode=args.scene_gate_mode,
                consecutive_fail_limit=max(1, args.scene_gate_consecutive_fails),
            )
        else:
            print("Court scene gate disabled; detection will continue through cutaways.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        detector = load_ball_detector(
            model_path=args.model_path,
            model_url=args.model_url,
            allow_download=not args.no_model_download,
        )
        tracker = BallKalmanTracker(max_missing_frames=args.max_missing, max_jump_px=args.max_jump_px)
        motion_detector = MotionBallDetector()

        if args.mode == "hardware":
            serial_conn = open_serial_connection(args.serial_port, args.baudrate)
            print("Hardware mode enabled: serial commands will be sent every processed frame.")
        else:
            print("Simulation mode enabled: no serial commands will be sent.")
        if args.topdown_background == "synthetic":
            print("Top-down visualization uses a synthetic court to avoid net warp distortion.")
        else:
            print("Top-down visualization uses warped video; vertical objects such as the net will distort.")

        ensure_parent_dir(args.output_video)
        ensure_parent_dir(args.csv_log)
        csv_file = open(args.csv_log, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_sec", "frame", "grid_row", "grid_col", "motor_id", "confidence"])

        output_fps = max(1.0, source_fps / float(args.frame_stride))
        last_valid_grid: Optional[GridState] = None
        motor_debounce_state = DebouncedMotorState()
        last_console_motor: Optional[int] = None
        fps_estimate = 0.0

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing", unit="frame")
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            pbar.update(1)

            if frame_index % args.frame_stride != 0:
                frame_index += 1
                continue

            start_time = time.perf_counter()
            timestamp_sec = frame_index / source_fps

            scene_visible = True
            scene_scores: Optional[tuple[float, float]] = None
            if scene_gate is not None:
                scene_visible, color_ratio, hist_similarity = scene_gate.score(frame)
                scene_scores = (color_ratio, hist_similarity)

            if scene_visible:
                observation = detect_and_track_ball(
                    frame=frame,
                    detector=detector,
                    tracker=tracker,
                    motion_detector=motion_detector,
                    conf_threshold=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    use_hsv_fallback=not args.disable_hsv_fallback,
                    use_motion_fallback=not args.disable_motion_fallback,
                    hsv_only=args.hsv_only,
                    court_mask=detection_roi.mask,
                )

                ball_xy = observation.center if observation is not None else None
                warped_frame, topdown_xy = apply_homography(frame, homography, ball_xy)
                raw_grid_state = map_to_motor(topdown_xy, last_valid_grid, args.outside_behavior)

                if raw_grid_state.inside:
                    last_valid_grid = raw_grid_state

                motor_id_to_send = debounce_motor_id(
                    raw_grid_state.motor_id,
                    motor_debounce_state,
                    args.debounce_ms,
                )
            else:
                tracker.reset()
                motion_detector.reset()
                observation = None
                warped_frame, topdown_xy = apply_homography(frame, homography, None)
                raw_grid_state = GridState(row=-1, col=-1, motor_id=-1, inside=False)
                motor_id_to_send = -1
                motor_debounce_state.current_motor_id = -1
                motor_debounce_state.last_change_time = time.perf_counter()

            # Log the detected/mapped ball cell. If debounce holds a previous motor,
            # the visualization and CSV still show the current ball-mapped cell.
            confidence = observation.confidence if observation is not None else 0.0
            csv_writer.writerow(
                [
                    f"{timestamp_sec:.3f}",
                    frame_index,
                    raw_grid_state.row,
                    raw_grid_state.col,
                    raw_grid_state.motor_id,
                    f"{confidence:.3f}",
                ]
            )

            if args.mode == "hardware":
                send_to_hardware(serial_conn, motor_id_to_send)

            if motor_id_to_send != last_console_motor:
                print(f"-> Motor {motor_id_to_send} activated")
                last_console_motor = motor_id_to_send

            if motor_id_to_send != last_console_motor:
                print(f"→ Motor {motor_id_to_send} activated")
                last_console_motor = motor_id_to_send

            original_vis = draw_original_overlay(
                frame,
                observation,
                court_points,
                detection_roi_points=detection_roi.points,
                scene_visible=scene_visible,
                scene_scores=scene_scores,
            )
            topdown_vis = draw_topdown_overlay(
                warped_frame,
                topdown_xy,
                raw_grid_state,
                background_mode=args.topdown_background,
            )

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            instant_fps = 1.0 / elapsed
            fps_estimate = instant_fps if fps_estimate <= 0 else (0.90 * fps_estimate + 0.10 * instant_fps)

            combined = make_side_by_side(
                original_vis=original_vis,
                topdown_vis=topdown_vis,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                motor_id=motor_id_to_send,
                fps_estimate=fps_estimate,
            )

            if video_writer is None:
                out_h, out_w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.output_video, fourcc, output_fps, (out_w, out_h))
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not open output video writer: {args.output_video}")
                print(f"Saving visualization video: {args.output_video} ({out_w}x{out_h} @ {output_fps:.2f} FPS)")

            video_writer.write(combined)

            if not args.no_display:
                if args.display_scale > 0 and abs(args.display_scale - 1.0) > 1e-3:
                    display = cv2.resize(
                        combined,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    display = combined
                cv2.imshow("Tennis haptic 6x10 - press q to quit", display)

                wait_ms = max(1, int(1000.0 / output_fps))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    print("Quit requested by user.")
                    break

            frame_index += 1

        print(f"CSV log saved: {args.csv_log}")
        print(f"Visualization video saved: {args.output_video}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if args.mode == "hardware" and serial_conn is not None:
            try:
                send_to_hardware(serial_conn, -1)
                time.sleep(0.05)
                serial_conn.close()
                print("Serial closed; sent M-1 all-off command.")
            except Exception as exc:
                print(f"WARNING: serial shutdown failed: {exc}")

        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()
        cap.release()
        if pbar is not None:
            pbar.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
