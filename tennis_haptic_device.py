
"""
Visually impaired tennis haptic feedback device (Upgraded with Shape/Motion Filters)

Pipeline:
    tennis match video -> tennis ball detection/tracking -> court homography
    -> 6x10 top-down grid -> motor id 0..59 -> optional USB serial command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =============================================================================
# Constants
# =============================================================================

DEFAULT_VIDEO_PATH = "tennis_match.mp4"
DEFAULT_OUTPUT_VIDEO_PATH = "tennis_haptic_6x10.mp4"
DEFAULT_CSV_LOG_PATH = "ball_grid_log.csv"

TOPDOWN_WIDTH = 1000
TOPDOWN_HEIGHT = 600
GRID_ROWS = 6
GRID_COLS = 10
CELL_WIDTH = TOPDOWN_WIDTH // GRID_COLS
CELL_HEIGHT = TOPDOWN_HEIGHT // GRID_ROWS

COURT_POINT_LABELS = [
    "1 near left baseline corner",
    "2 near right baseline corner",
    "3 far right sideline corner",
    "4 far left sideline corner",
]

MAX_SELECTION_WINDOW_WIDTH = 1400
MAX_SELECTION_WINDOW_HEIGHT = 900

# YOLO
DEFAULT_YOLO_MODEL_PATH = "models/tennis_ball_yolov8_best.pt"
GENERIC_YOLO_FALLBACK_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_YOLO_IMAGE_SIZE = 640
YOLO_MAX_BOX_FRACTION = 0.18

# HSV is only an auxiliary signal in hybrid mode. It should help the scorer
# prefer tennis-ball-like candidates, not become the only detector.
HSV_RANGES = [
    ((22, 80, 120), (45, 255, 255)),
    ((46, 70, 120), (78, 255, 255)),
]
HSV_MIN_CONTOUR_AREA = 6
HSV_MAX_CONTOUR_AREA = 2500
HSV_MIN_CIRCULARITY = 0.35
HSV_MIN_ASPECT_RATIO = 0.35
HSV_MAX_ASPECT_RATIO = 2.80
HSV_BLUR_MIN_CIRCULARITY = 0.08
HSV_BLUR_MIN_ASPECT_RATIO = 0.14
HSV_BLUR_MAX_ASPECT_RATIO = 7.00
HSV_BLUR_MAX_CONTOUR_AREA = 900
HSV_BLUR_MIN_YELLOW_SCORE = 0.10
YELLOW_SCORE_WEIGHT = 0.22

# Very fast tennis balls can become a thin motion streak instead of an ellipse.
# These constraints keep long racket/player parts out while allowing small,
# yellow-green streaks anywhere inside the calibrated court ROI.
STREAK_HSV_RANGES = [
    ((18, 45, 75), (48, 255, 255)),
    ((49, 35, 70), (88, 255, 255)),
]
DEFAULT_STREAK_MIN_YELLOW_SCORE = 0.08
STREAK_MIN_LENGTH = 10
STREAK_MAX_LENGTH = 180
STREAK_MAX_THICKNESS = 22
STREAK_MIN_ASPECT_RATIO = 3.0
STREAK_MAX_AREA = 1200
STREAK_MIN_FILL_RATIO = 0.16

# Player suppression removes false ball candidates on clothing/shoes.
DEFAULT_PLAYER_SUPPRESSION = True
DEFAULT_PLAYER_MODEL_PATH = GENERIC_YOLO_FALLBACK_MODEL
DEFAULT_PLAYER_CONFIDENCE = 0.12
DEFAULT_PLAYER_DETECT_STRIDE = 1
DEFAULT_PLAYER_BOX_MARGIN = 95
DEFAULT_PLAYER_OVERLAP_THRESHOLD = 0.10

# Roundness gate: reject elongated racket/shoe/wristband candidates.
DEFAULT_REQUIRE_ROUND_CANDIDATE = True
DEFAULT_ROUNDNESS_THRESHOLD = 0.42
DEFAULT_YOLO_ROUNDNESS_THRESHOLD = 0.30
ROUND_ASPECT_MIN = 0.55
ROUND_ASPECT_MAX = 1.85
ROUNDNESS_YELLOW_RELAXATION = 0.08
BLUR_ELLIPSE_MIN_YELLOW_SCORE = 0.10
BLUR_ELLIPSE_MIN_ASPECT_RATIO = 0.14
BLUR_ELLIPSE_MAX_ASPECT_RATIO = 18.00
BLUR_ELLIPSE_MAX_AREA = 900
BLUR_ELLIPSE_MIN_ROUNDNESS = 0.0

# Motion candidates are the main source of white shoe/racket/wristband false
# positives. If a moving blob has no yellow evidence, it must look much more
# ball-like and stay small.
MOTION_MAX_BOX_SIDE = 130
MOTION_MIN_ASPECT_RATIO = 0.14
MOTION_MAX_ASPECT_RATIO = 18.00
MOTION_LOW_YELLOW_SCORE = 0.06
DEFAULT_MOTION_MIN_YELLOW_SCORE = 0.08
MOTION_LOW_YELLOW_MIN_CIRCULARITY = 0.35
MOTION_LOW_YELLOW_MAX_AREA = 350

# Generic COCO sports-ball YOLO can confuse white shoes/rackets with a ball.
DEFAULT_GENERIC_YOLO_MIN_YELLOW_SCORE = 0.10
GENERIC_YOLO_HIGH_CONF_OVERRIDE = 0.88

# Colors
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
    center: tuple[float, float]
    confidence: float
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    source: str
    yellow_score: float = 0.0
    roundness_score: float = 0.0


@dataclass
class BallObservation:
    center: Optional[tuple[float, float]]
    confidence: float
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    source: str
    predicted: bool


@dataclass
class DetectorBundle:
    model: Any
    model_path: Optional[str]
    source_name: str


@dataclass
class RoiSelection:
    mask: Optional[np.ndarray]
    points: Optional[np.ndarray]
    source: str


# =============================================================================
# Helper Functions
# =============================================================================

def draw_text(image, text, org, scale=0.65, color=COLOR_WHITE, thickness=2):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """비율을 유지하면서 이미지의 높이를 지정된 크기로 변경합니다."""
    h, w = image.shape[:2]
    if h == target_height:
        return image
    scale = target_height / h
    target_width = int(w * scale)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def is_ball_class(names, class_id):
    if class_id == -1:
        return False
    if isinstance(names, dict) and len(names) == 1:
        return True
    name = str(names.get(class_id, "")).lower() if isinstance(names, dict) else ""
    return ("ball" in name) or ("tennis" in name) or ("sports ball" in name)


def compute_yellow_score(frame, bbox_xyxy):
    if bbox_xyxy is None:
        return 0.0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = int(np.clip(x1 - 3, 0, w - 1))
    y1 = int(np.clip(y1 - 3, 0, h - 1))
    x2 = int(np.clip(x2 + 3, x1 + 1, w))
    y2 = int(np.clip(y2 + 3, y1 + 1, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yellow_pixels = 0
    for lower, upper in HSV_RANGES:
        yellow_pixels += int(cv2.countNonZero(cv2.inRange(hsv, np.array(lower), np.array(upper))))
    return float(np.clip(yellow_pixels / float(max(patch.shape[0] * patch.shape[1], 1)), 0.0, 1.0))


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_roundness_score(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / float(h)
    if aspect < ROUND_ASPECT_MIN or aspect > ROUND_ASPECT_MAX:
        return 0.0
    return min(w, h) / float(max(w, h))


def bbox_aspect_and_area(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / float(h)
    normalized_aspect = min(aspect, 1.0 / max(aspect, 1e-6))
    area = w * h
    return aspect, normalized_aspect, area


def contour_streak_metrics(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _angle = rect
    length = max(float(rw), float(rh))
    thickness = max(1.0, min(float(rw), float(rh)))
    aspect = length / thickness
    area = float(cv2.contourArea(contour))
    fill_ratio = area / max(length * thickness, 1.0)
    return (float(cx), float(cy)), length, thickness, aspect, area, fill_ratio


def is_motion_blur_ellipse(candidate):
    if candidate.bbox_xyxy is None:
        return False
    if candidate.yellow_score < BLUR_ELLIPSE_MIN_YELLOW_SCORE:
        return False
    aspect, normalized_aspect, area = bbox_aspect_and_area(candidate.bbox_xyxy)
    elongated_ok = BLUR_ELLIPSE_MIN_ASPECT_RATIO <= normalized_aspect <= 1.0
    long_side_aspect_ok = (
        BLUR_ELLIPSE_MIN_ASPECT_RATIO <= aspect <= BLUR_ELLIPSE_MAX_ASPECT_RATIO
        or BLUR_ELLIPSE_MIN_ASPECT_RATIO <= (1.0 / max(aspect, 1e-6)) <= BLUR_ELLIPSE_MAX_ASPECT_RATIO
    )
    return elongated_ok and long_side_aspect_ok and area <= BLUR_ELLIPSE_MAX_AREA


def contour_roundness_score(contour, bbox):
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    circularity = 0.0
    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    aspect_score = bbox_roundness_score(bbox)
    return float(np.clip(0.55 * aspect_score + 0.45 * min(circularity, 1.0), 0.0, 1.0))


def estimate_visual_roundness(frame, bbox):
    """
    Estimate roundness from bbox plus local contour shape.

    This is intentionally lightweight. It rejects elongated racket/shoe/wrist
    parts while allowing small tennis balls that are not perfectly circular.
    """

    if bbox is None:
        return 0.0
    aspect_score = bbox_roundness_score(bbox)
    if aspect_score <= 0:
        return 0.0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = 4
    x1 = int(np.clip(x1 - pad, 0, w - 1))
    y1 = int(np.clip(y1 - pad, 0, h - 1))
    x2 = int(np.clip(x2 + pad, x1 + 1, w))
    y2 = int(np.clip(y2 + pad, y1 + 1, h))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return aspect_score

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return aspect_score

    largest = max(contours, key=cv2.contourArea)
    contour_score = contour_roundness_score(largest, bbox)
    return float(np.clip(0.45 * aspect_score + 0.55 * contour_score, 0.0, 1.0))


def filter_round_candidates(candidates, frame, threshold, yolo_threshold):
    filtered = []
    rejected = 0
    for candidate in candidates:
        roundness = candidate.roundness_score
        if roundness <= 0 and candidate.bbox_xyxy is not None:
            roundness = estimate_visual_roundness(frame, candidate.bbox_xyxy)
            candidate.roundness_score = roundness

        effective_threshold = yolo_threshold if candidate.source in ("YOLO", "YOLO_GENERIC") else threshold
        if candidate.yellow_score >= 0.20:
            effective_threshold = max(0.15, effective_threshold - ROUNDNESS_YELLOW_RELAXATION)

        if candidate.source == "Streak":
            filtered.append(candidate)
        elif is_motion_blur_ellipse(candidate) and roundness >= BLUR_ELLIPSE_MIN_ROUNDNESS:
            filtered.append(candidate)
        elif roundness >= effective_threshold:
            filtered.append(candidate)
        else:
            rejected += 1
    return filtered, rejected


def filter_accessory_candidates(candidates, generic_yolo_min_yellow, motion_min_yellow):
    """
    Remove common white accessory false positives.

    Wristbands, headbands, socks, shoes, and racket heads can be round-ish and
    moving. When they come from generic COCO sports-ball or motion detection,
    require tennis-ball color evidence unless the generic YOLO confidence is
    extremely high.
    """

    filtered = []
    rejected = 0
    for candidate in candidates:
        if candidate.source == "YOLO_GENERIC":
            if (
                candidate.yellow_score < generic_yolo_min_yellow
                and candidate.confidence < GENERIC_YOLO_HIGH_CONF_OVERRIDE
            ):
                rejected += 1
                continue
        elif candidate.source in ("Motion", "Streak"):
            if candidate.yellow_score < motion_min_yellow:
                rejected += 1
                continue
        filtered.append(candidate)
    return filtered, rejected


def bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def expand_bbox(bbox, margin, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        int(np.clip(x1 - margin, 0, w - 1)),
        int(np.clip(y1 - margin, 0, h - 1)),
        int(np.clip(x2 + margin, 0, w - 1)),
        int(np.clip(y2 + margin, 0, h - 1)),
    )


def point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def filter_candidates_inside_players(candidates, player_boxes, overlap_threshold):
    if not player_boxes:
        return candidates

    filtered = []
    for candidate in candidates:
        suppress = False
        for player_box in player_boxes:
            if point_inside_bbox(candidate.center, player_box):
                suppress = True
                break

            if candidate.bbox_xyxy is not None:
                overlap = bbox_intersection_area(candidate.bbox_xyxy, player_box)
                ratio = overlap / float(max(bbox_area(candidate.bbox_xyxy), 1))
                if ratio >= overlap_threshold:
                    suppress = True
                    break

        if not suppress:
            filtered.append(candidate)
    return filtered


def select_court_points(frame: np.ndarray) -> np.ndarray:
    original_h, original_w = frame.shape[:2]
    scale = min(1.0, MAX_SELECTION_WINDOW_WIDTH / float(original_w), MAX_SELECTION_WINDOW_HEIGHT / float(original_h))
    display_w = int(round(original_w * scale))
    display_h = int(round(original_h * scale))
    display_base = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)

    points = []
    window_name = "Select 4 court corners - r reset, q quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nCourt calibration: click exactly 4 visible court corners.")
    while True:
        canvas = display_base.copy()
        for idx, (px, py) in enumerate(points):
            sx, sy = int(round(px * scale)), int(round(py * scale))
            cv2.circle(canvas, (sx, sy), 7, COLOR_RED, -1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            sys.exit(0)
        if len(points) == 4 and key in (13, 10, 32):
            break
        if len(points) == 4:
            cv2.waitKey(400)
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


def build_homography(src_points: np.ndarray) -> np.ndarray:
    dst_points = np.array([
        [0, TOPDOWN_HEIGHT - 1], [TOPDOWN_WIDTH - 1, TOPDOWN_HEIGHT - 1],
        [TOPDOWN_WIDTH - 1, 0], [0, 0]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)


def make_court_roi_mask(frame_shape, court_points, margin_px=80):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = court_points.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px, margin_px))
    return cv2.dilate(mask, kernel, iterations=1)


# =============================================================================
# Detectors (Upgraded with Filters)
# =============================================================================

def load_ball_detector(model_path):
    try:
        from ultralytics import YOLO
        resolved_model = model_path
        if not resolved_model and Path(DEFAULT_YOLO_MODEL_PATH).exists():
            resolved_model = DEFAULT_YOLO_MODEL_PATH
        if not resolved_model:
            resolved_model = GENERIC_YOLO_FALLBACK_MODEL
        model = YOLO(resolved_model)
        print(f"Loaded YOLO model: {resolved_model}")
        return DetectorBundle(model=model, model_path=resolved_model, source_name="YOLO")
    except Exception as exc:
        print(f"YOLO unavailable ({exc}). Falling back to HSV/Motion filters.")
        return DetectorBundle(model=None, model_path=None, source_name="Filters-only")


def detect_yolo_candidates(frame, detector, conf_threshold, court_mask=None):
    if detector.model is None: return []
    results = detector.model.predict(source=frame, conf=conf_threshold, verbose=False)
    candidates = []
    if not results or not getattr(results[0], "boxes", None): return candidates

    boxes = results[0].boxes
    names = getattr(results[0], "names", getattr(detector.model, "names", {}))
    for box in boxes:
        conf = float(box.conf[0].cpu().numpy())
        if conf < conf_threshold: continue
        cls_id = int(box.cls[0].cpu().numpy()) if getattr(box, "cls", None) is not None else -1
        if names and not is_ball_class(names, cls_id):
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0

        if court_mask is not None:
            ix = int(np.clip(round(cx), 0, frame.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, frame.shape[0] - 1))
            if court_mask[iy, ix] == 0: continue

        bbox = tuple(int(v) for v in xyxy)
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        if max(bw, bh) > max(frame.shape[:2]) * YOLO_MAX_BOX_FRACTION:
            continue

        yellow_score = compute_yellow_score(frame, bbox)
        is_generic_model = Path(str(detector.model_path)).name == GENERIC_YOLO_FALLBACK_MODEL
        source = "YOLO_GENERIC" if is_generic_model else "YOLO"

        candidates.append(DetectionCandidate(
            center=(cx, cy), confidence=conf, bbox_xyxy=bbox, source=source,
            yellow_score=yellow_score,
            roundness_score=estimate_visual_roundness(frame, bbox)
        ))
    return candidates


def detect_hsv_candidates(frame: np.ndarray, conf_threshold: float, court_mask: Optional[np.ndarray] = None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for lower, upper in HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

    if court_mask is not None:
        mask = cv2.bitwise_and(mask, court_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < HSV_MIN_CONTOUR_AREA or area > HSV_MAX_CONTOUR_AREA:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(float(h), 1.0)
        bbox = (x, y, x + w, y + h)
        yellow_score = compute_yellow_score(frame, bbox)

        normal_ball_shape = (
            circularity >= HSV_MIN_CIRCULARITY
            and HSV_MIN_ASPECT_RATIO <= aspect <= HSV_MAX_ASPECT_RATIO
        )
        blur_ball_shape = (
            yellow_score >= HSV_BLUR_MIN_YELLOW_SCORE
            and circularity >= HSV_BLUR_MIN_CIRCULARITY
            and area <= HSV_BLUR_MAX_CONTOUR_AREA
            and (
                HSV_BLUR_MIN_ASPECT_RATIO <= aspect <= HSV_BLUR_MAX_ASPECT_RATIO
                or HSV_BLUR_MIN_ASPECT_RATIO <= (1.0 / max(aspect, 1e-6)) <= HSV_BLUR_MAX_ASPECT_RATIO
            )
        )
        if not (normal_ball_shape or blur_ball_shape):
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

        area_score = min(area / 80.0, 1.0)
        blur_bonus = 0.10 if blur_ball_shape and not normal_ball_shape else 0.0
        confidence = float(np.clip(0.35 + 0.40 * min(circularity, 1.0) + 0.25 * area_score + blur_bonus, 0.0, 0.92))
        if confidence < conf_threshold:
            continue

        roundness = contour_roundness_score(contour, bbox)
        candidates.append(DetectionCandidate(
            center=(float(cx), float(cy)), confidence=confidence,
            bbox_xyxy=bbox, source="HSV",
            yellow_score=yellow_score,
            roundness_score=roundness
        ))

    return candidates


def detect_streak_candidates(
    frame: np.ndarray,
    conf_threshold: float,
    court_mask: Optional[np.ndarray] = None,
    min_yellow_score: float = DEFAULT_STREAK_MIN_YELLOW_SCORE,
):
    """Detect fast tennis-ball motion streaks using elongated yellow-green contours."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in STREAK_HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

    if court_mask is not None:
        mask = cv2.bitwise_and(mask, court_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        center, length, thickness, streak_aspect, area, fill_ratio = contour_streak_metrics(contour)
        if area < HSV_MIN_CONTOUR_AREA or area > STREAK_MAX_AREA:
            continue
        if length < STREAK_MIN_LENGTH or length > STREAK_MAX_LENGTH:
            continue
        if thickness > STREAK_MAX_THICKNESS:
            continue
        if streak_aspect < STREAK_MIN_ASPECT_RATIO:
            continue
        if fill_ratio < STREAK_MIN_FILL_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, x + w, y + h)
        yellow_score = compute_yellow_score(frame, bbox)
        if yellow_score < min_yellow_score:
            continue

        # Long, thin, yellow-green streaks are likely fast balls. Keep confidence
        # modest so YOLO/normal HSV detections still win when available.
        length_score = min(length / 80.0, 1.0)
        aspect_score = min(streak_aspect / 8.0, 1.0)
        thin_score = 1.0 - min(thickness / float(STREAK_MAX_THICKNESS), 1.0)
        confidence = float(np.clip(
            0.34 + 0.20 * yellow_score + 0.18 * length_score + 0.16 * aspect_score + 0.10 * thin_score,
            0.0,
            0.88,
        ))
        if confidence < conf_threshold:
            continue

        candidates.append(DetectionCandidate(
            center=center,
            confidence=confidence,
            bbox_xyxy=bbox,
            source="Streak",
            yellow_score=yellow_score,
            roundness_score=0.0,
        ))

    return candidates


def detect_shape_candidates(frame: np.ndarray, conf_threshold: float, court_mask: Optional[np.ndarray] = None) -> list[
    DetectionCandidate]:
    """ [업그레이드] 색상 필터 대신 허프 원 변환(Hough Circle) 형태 기반 필터 적용 """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    if court_mask is not None:
        gray = cv2.bitwise_and(gray, court_mask)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
        param1=50, param2=25, minRadius=2, maxRadius=25
    )

    candidates = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            confidence = 0.65
            if confidence < conf_threshold: continue
            bbox = (x - r, y - r, x + r, y + r)
            candidates.append(DetectionCandidate(
                center=(float(x), float(y)), confidence=confidence,
                bbox_xyxy=bbox, source="ShapeFilter"
            ))
    return candidates


class PureMotionBallDetector:
    """ [업그레이드] 색상을 배제하고 순수 MOG2 움직임 분리 및 형태 분석 필터 적용 """

    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)

    def detect(self, frame: np.ndarray, conf_threshold: float, court_mask: Optional[np.ndarray] = None):
        fg_mask = self.back_sub.apply(frame)
        if court_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, court_mask)

        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 4 or area > 1800: continue

            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) > MOTION_MAX_BOX_SIDE:
                continue
            aspect = w / max(float(h), 1.0)
            if not (MOTION_MIN_ASPECT_RATIO <= aspect <= MOTION_MAX_ASPECT_RATIO):
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            bbox = (x, y, x + w, y + h)
            yellow_score = compute_yellow_score(frame, bbox)
            if circularity < 0.10 and yellow_score < BLUR_ELLIPSE_MIN_YELLOW_SCORE:
                continue

            if yellow_score < MOTION_LOW_YELLOW_SCORE:
                if circularity < MOTION_LOW_YELLOW_MIN_CIRCULARITY:
                    continue
                if area > MOTION_LOW_YELLOW_MAX_AREA:
                    continue

            cx, cy = x + w / 2, y + h / 2
            area_score = min(area / 120.0, 1.0)
            yellow_bonus = 0.12 if yellow_score >= MOTION_LOW_YELLOW_SCORE else -0.08
            confidence = float(np.clip(0.4 + 0.3 * circularity + 0.2 * area_score + yellow_bonus, 0.0, 0.85))

            if confidence < conf_threshold: continue
            candidates.append(DetectionCandidate(
                center=(float(cx), float(cy)), confidence=confidence,
                bbox_xyxy=bbox, source="Motion",
                yellow_score=yellow_score,
                roundness_score=contour_roundness_score(contour, bbox)
            ))
        return candidates


class PlayerSuppressor:
    """Detect player boxes and remove ball candidates on player clothing/shoes."""

    def __init__(
        self,
        enabled=True,
        model_path=DEFAULT_PLAYER_MODEL_PATH,
        confidence=DEFAULT_PLAYER_CONFIDENCE,
        stride=DEFAULT_PLAYER_DETECT_STRIDE,
        margin=DEFAULT_PLAYER_BOX_MARGIN,
    ):
        self.enabled = enabled
        self.confidence = confidence
        self.stride = max(1, stride)
        self.margin = margin
        self.frame_counter = 0
        self.model = None
        self.boxes = []

        if not enabled:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Loaded player suppressor model: {model_path}")
        except Exception as exc:
            print(f"Player suppression disabled: could not load person detector ({exc})")
            self.enabled = False

    def update(self, frame, court_mask=None):
        if not self.enabled or self.model is None:
            return []

        self.frame_counter += 1
        if self.frame_counter > 1 and (self.frame_counter % self.stride) != 0:
            return self.boxes

        try:
            results = self.model.predict(source=frame, conf=self.confidence, classes=[0], verbose=False)
        except Exception as exc:
            print(f"WARNING: player detector failed: {exc}")
            return self.boxes

        boxes = []
        if results and getattr(results[0], "boxes", None):
            for box in results[0].boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf < self.confidence:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                # Keep only players that plausibly overlap the calibrated court.
                if court_mask is not None:
                    cx = int(np.clip((x1 + x2) / 2, 0, frame.shape[1] - 1))
                    foot_y = int(np.clip(y2, 0, frame.shape[0] - 1))
                    center_y = int(np.clip((y1 + y2) / 2, 0, frame.shape[0] - 1))
                    if court_mask[foot_y, cx] == 0 and court_mask[center_y, cx] == 0:
                        continue

                boxes.append(expand_bbox((x1, y1, x2, y2), self.margin, frame.shape))

        self.boxes = boxes
        return self.boxes


def score_candidate(candidate: DetectionCandidate) -> float:
    source_weight = {
        "YOLO": 1.00,
        "YOLO_GENERIC": 0.82,
        "HSV": 0.78,
        "Streak": 0.76,
        "Motion": 0.70,
    }.get(candidate.source, 0.65)
    if candidate.source == "Streak":
        blur_bonus = 0.20
    else:
        blur_bonus = 0.14 if is_motion_blur_ellipse(candidate) else 0.0
    return (
        candidate.confidence * source_weight
        + YELLOW_SCORE_WEIGHT * candidate.yellow_score
        + 0.18 * candidate.roundness_score
        + blur_bonus
    )


# =============================================================================
# SofaScore Live Tennis Text Output
# =============================================================================

SOFASCORE_BASE_URLS = (
    "https://api.sofascore.com/api/v1",
    "https://www.sofascore.com/api/v1",
)


@dataclass
class TennisEventSelection:
    event_id: int
    title: str
    tournament: str


class SofascoreClient:
    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://www.sofascore.com/tennis",
        }

    def get_json(self, path: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for base_url in SOFASCORE_BASE_URLS:
            url = f"{base_url}{path}"
            try:
                return self._get_json_with_urllib(url)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                try:
                    return self._get_json_with_curl_cffi(url)
                except RuntimeError as fallback_exc:
                    last_error = fallback_exc

        raise RuntimeError(
            "SofaScore 정보를 불러오지 못했습니다. "
            "사이트에서 비브라우저 요청을 차단할 수 있습니다. "
            f"마지막 오류: {last_error}"
        )

    def _get_json_with_urllib(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)

    def _get_json_with_curl_cffi(self, url: str) -> dict[str, Any]:
        """
        Optional fallback for environments where SofaScore blocks urllib.

        Install only if needed:
            pip install curl_cffi
        """

        try:
            from curl_cffi import requests as curl_requests
        except Exception as exc:
            raise RuntimeError(f"curl_cffi fallback을 사용할 수 없습니다: {exc}") from exc

        response = curl_requests.get(
            url,
            headers=self.headers,
            timeout=self.timeout,
            impersonate="chrome",
        )
        if response.status_code >= 400:
            raise RuntimeError(f"curl_cffi 요청 실패: HTTP {response.status_code}")
        return dict(response.json())

    def live_tennis_events(self) -> list[dict[str, Any]]:
        data = self.get_json("/sport/tennis/events/live")
        return list(data.get("events", []))

    def event_detail(self, event_id: int) -> dict[str, Any]:
        data = self.get_json(f"/event/{event_id}")
        return dict(data.get("event", data))

    def event_statistics(self, event_id: int) -> dict[str, Any]:
        return self.get_json(f"/event/{event_id}/statistics")


def clean_text(value: Any, default: str = "정보 없음") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def team_name(team: dict[str, Any]) -> str:
    return clean_text(team.get("name") or team.get("shortName"))


def event_title(event: dict[str, Any]) -> str:
    home = team_name(event.get("homeTeam", {}))
    away = team_name(event.get("awayTeam", {}))
    return f"{home} 대 {away}"


def tournament_name(event: dict[str, Any]) -> str:
    tournament = event.get("tournament", {}) or {}
    unique_tournament = tournament.get("uniqueTournament", {}) or {}
    return clean_text(unique_tournament.get("name") or tournament.get("name"))


def status_text(event: dict[str, Any]) -> str:
    status = event.get("status", {}) or {}
    description = clean_text(status.get("description"), "상태 정보 없음")
    status_type = clean_text(status.get("type"), "")
    if status_type:
        return f"{description} ({status_type})"
    return description


def score_value(score: dict[str, Any], key: str) -> str | None:
    value = score.get(key)
    if value is None:
        return None
    return str(value)


def list_set_keys(home_score: dict[str, Any], away_score: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for index in range(1, 6):
        key = f"period{index}"
        if key in home_score or key in away_score:
            keys.append(key)
    return keys


def format_set_scores(event: dict[str, Any]) -> str:
    home = team_name(event.get("homeTeam", {}))
    away = team_name(event.get("awayTeam", {}))
    home_score = event.get("homeScore", {}) or {}
    away_score = event.get("awayScore", {}) or {}
    set_keys = list_set_keys(home_score, away_score)

    lines = [f"세트별 스코어입니다. {home} 대 {away}."]
    if not set_keys:
        lines.append("아직 세트별 스코어 정보가 없습니다.")
    for key in set_keys:
        set_number = key.replace("period", "")
        home_value = score_value(home_score, key) or "0"
        away_value = score_value(away_score, key) or "0"
        lines.append(f"{set_number}세트: {home} {home_value}, {away} {away_value}.")

        home_tiebreak = score_value(home_score, f"{key}TieBreak")
        away_tiebreak = score_value(away_score, f"{key}TieBreak")
        if home_tiebreak is not None or away_tiebreak is not None:
            lines.append(
                f"{set_number}세트 타이브레이크: "
                f"{home} {home_tiebreak or '0'}, {away} {away_tiebreak or '0'}."
            )

    home_point = score_value(home_score, "point")
    away_point = score_value(away_score, "point")
    if home_point is not None or away_point is not None:
        lines.append(f"현재 포인트: {home} {home_point or '0'}, {away} {away_point or '0'}.")

    home_match_score = score_value(home_score, "current")
    away_match_score = score_value(away_score, "current")
    if home_match_score is not None or away_match_score is not None:
        lines.append(f"현재 세트 스코어: {home} {home_match_score or '0'}, {away} {away_match_score or '0'}.")

    return "\n".join(lines)


def player_entries(team: dict[str, Any]) -> Iterable[dict[str, Any]]:
    sub_teams = team.get("subTeams")
    if isinstance(sub_teams, list) and sub_teams:
        for player in sub_teams:
            if isinstance(player, dict):
                yield player
    else:
        yield team


def format_ranking(value: Any) -> str:
    if value is None:
        return "랭킹 정보 없음"
    return f"{value}위"


def format_player_info(event: dict[str, Any]) -> str:
    home_team = event.get("homeTeam", {}) or {}
    away_team = event.get("awayTeam", {}) or {}
    lines = [f"선수 정보입니다. 경기: {event_title(event)}."]

    for label, team in (("홈", home_team), ("어웨이", away_team)):
        lines.append(f"{label} 선수:")
        for player in player_entries(team):
            name = clean_text(player.get("name") or player.get("shortName"))
            ranking = format_ranking(player.get("ranking"))
            country = clean_text((player.get("country") or {}).get("name"), "")
            if country:
                lines.append(f"{name}, 국가 {country}, 현재 랭킹 {ranking}.")
            else:
                lines.append(f"{name}, 현재 랭킹 {ranking}.")

    return "\n".join(lines)


def statistics_periods(stats_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = stats_data.get("statistics", [])
    if isinstance(raw, dict):
        raw = [raw]
    return [period for period in raw if isinstance(period, dict)]


def format_statistics(stats_data: dict[str, Any], event: dict[str, Any]) -> str:
    home = team_name(event.get("homeTeam", {}))
    away = team_name(event.get("awayTeam", {}))
    periods = statistics_periods(stats_data)
    lines = [f"경기 세부 성적입니다. {home} 대 {away}."]

    if not periods:
        lines.append("현재 SofaScore에서 제공하는 statistics 정보가 없습니다.")
        return "\n".join(lines)

    for period in periods:
        period_name = clean_text(period.get("periodName") or period.get("period") or period.get("name"), "전체")
        lines.append(f"{period_name} 기준 성적:")
        groups = period.get("groups", [])
        if not isinstance(groups, list):
            continue

        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = clean_text(group.get("groupName") or group.get("name"), "")
            if group_name:
                lines.append(f"{group_name}:")

            items = group.get("statisticsItems", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = clean_text(item.get("name"))
                home_value = clean_text(item.get("home"), "0")
                away_value = clean_text(item.get("away"), "0")
                lines.append(f"{name}: {home} {home_value}, {away} {away_value}.")

    return "\n".join(lines)


def format_summary(event: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"선택한 경기는 {event_title(event)}입니다.",
            f"대회: {tournament_name(event)}.",
            f"경기 상태: {status_text(event)}.",
            format_set_scores(event),
        ]
    )


def build_output(event: dict[str, Any], detail: str, client: SofascoreClient) -> str:
    if detail == "summary":
        return format_summary(event)
    if detail == "players":
        return format_player_info(event)
    if detail == "sets":
        return format_set_scores(event)
    if detail == "stats":
        try:
            stats_data = client.event_statistics(int(event["id"]))
        except RuntimeError as exc:
            return f"경기 세부 성적을 불러오지 못했습니다. {exc}"
        return format_statistics(stats_data, event)
    if detail == "all":
        sections = [format_summary(event), format_player_info(event)]
        try:
            sections.append(format_statistics(client.event_statistics(int(event["id"])), event))
        except RuntimeError as exc:
            sections.append(f"경기 세부 성적을 불러오지 못했습니다. {exc}")
        return "\n\n".join(sections)
    raise ValueError(f"지원하지 않는 상세 정보 유형입니다: {detail}")


def event_matches_query(event: dict[str, Any], query: str) -> bool:
    query_normalized = query.casefold()
    fields = [
        event_title(event),
        tournament_name(event),
        clean_text(event.get("slug"), ""),
        clean_text(event.get("customId"), ""),
    ]
    return any(query_normalized in field.casefold() for field in fields)


def make_selection(event: dict[str, Any]) -> TennisEventSelection:
    return TennisEventSelection(
        event_id=int(event["id"]),
        title=event_title(event),
        tournament=tournament_name(event),
    )


def print_event_list(events: list[dict[str, Any]], limit: int) -> None:
    if not events:
        print("현재 진행 중인 테니스 경기가 없습니다.")
        return

    print("현재 진행 중인 테니스 경기 목록입니다.")
    for index, event in enumerate(events[:limit], start=1):
        selection = make_selection(event)
        print(f"{index}. [{selection.event_id}] {selection.title} / {selection.tournament} / {status_text(event)}")


def select_event_interactively(events: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    visible_events = events[:limit]
    print_event_list(visible_events, limit)
    if not visible_events:
        raise RuntimeError("선택할 수 있는 경기가 없습니다.")

    while True:
        user_input = input("경기 번호 또는 SofaScore event id를 입력하세요: ").strip()
        if not user_input:
            continue
        if user_input.isdigit():
            number = int(user_input)
            if 1 <= number <= len(visible_events):
                return visible_events[number - 1]
            for event in visible_events:
                if int(event.get("id", -1)) == number:
                    return event
        print("입력한 값과 일치하는 경기가 없습니다. 다시 입력해 주세요.")


def choose_detail_interactively() -> str:
    choices = {
        "1": "players",
        "2": "sets",
        "3": "stats",
        "4": "summary",
        "5": "all",
        "players": "players",
        "sets": "sets",
        "stats": "stats",
        "summary": "summary",
        "all": "all",
        "선수": "players",
        "세트": "sets",
        "성적": "stats",
        "요약": "summary",
        "전체": "all",
    }
    print("원하는 정보를 선택하세요.")
    print("1. 선수 이름과 랭킹")
    print("2. 세트별 스코어")
    print("3. 경기 내 세부 성적 statistics")
    print("4. 경기 요약")
    print("5. 전체")

    while True:
        user_input = input("요청 정보: ").strip().casefold()
        detail = choices.get(user_input)
        if detail:
            return detail
        print("지원하는 입력은 선수, 세트, 성적, 요약, 전체입니다.")


def resolve_event(args: argparse.Namespace, client: SofascoreClient) -> dict[str, Any]:
    if args.event_id is not None:
        return client.event_detail(args.event_id)

    events = client.live_tennis_events()
    if args.query:
        matches = [event for event in events if event_matches_query(event, args.query)]
        if not matches:
            raise RuntimeError(f"검색어 '{args.query}'와 일치하는 실시간 경기가 없습니다.")
        if args.non_interactive:
            return matches[0]
        return select_event_interactively(matches, args.limit)

    if args.non_interactive:
        if not events:
            raise RuntimeError("현재 진행 중인 테니스 경기가 없습니다.")
        return events[0]
    return select_event_interactively(events, args.limit)



def run_live_info_mode(args: argparse.Namespace) -> None:
    client = SofascoreClient()
    mapped_args = argparse.Namespace(
        event_id=args.live_event_id,
        query=args.live_query,
        non_interactive=args.live_non_interactive,
        limit=args.live_limit,
    )

    if args.live_list:
        print_event_list(client.live_tennis_events(), args.live_limit)
        return

    event = resolve_event(mapped_args, client)
    detail = args.live_detail or choose_detail_interactively()
    output = build_output(event, detail, client)
    print("\n===== OUTPUT =====")
    print(output)

# =============================================================================
# Main Pipeline & Serial
# =============================================================================

def send_to_hardware(serial_conn, motor_id):
    if serial_conn and serial_conn.is_open:
        if motor_id < 0:
            serial_conn.write(b"M-1\n")
        else:
            serial_conn.write(f"M{motor_id}\n".encode('ascii'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--mode", choices=["sim", "hardware"], default="sim")
    parser.add_argument("--serial_port", default="")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--disable_player_suppression", action="store_true")
    parser.add_argument("--player_model_path", default=DEFAULT_PLAYER_MODEL_PATH)
    parser.add_argument("--player_conf", type=float, default=DEFAULT_PLAYER_CONFIDENCE)
    parser.add_argument("--player_stride", type=int, default=DEFAULT_PLAYER_DETECT_STRIDE)
    parser.add_argument("--player_margin", type=int, default=DEFAULT_PLAYER_BOX_MARGIN)
    parser.add_argument("--player_overlap", type=float, default=DEFAULT_PLAYER_OVERLAP_THRESHOLD)
    parser.add_argument("--disable_roundness_gate", action="store_true")
    parser.add_argument("--roundness_threshold", type=float, default=DEFAULT_ROUNDNESS_THRESHOLD)
    parser.add_argument("--yolo_roundness_threshold", type=float, default=DEFAULT_YOLO_ROUNDNESS_THRESHOLD)
    parser.add_argument("--disable_accessory_filter", action="store_true")
    parser.add_argument("--generic_yolo_min_yellow", type=float, default=DEFAULT_GENERIC_YOLO_MIN_YELLOW_SCORE)
    parser.add_argument("--motion_min_yellow", type=float, default=DEFAULT_MOTION_MIN_YELLOW_SCORE)
    parser.add_argument("--disable_streak_detector", action="store_true")
    parser.add_argument("--streak_min_yellow", type=float, default=DEFAULT_STREAK_MIN_YELLOW_SCORE)

    parser.add_argument("--live_info", action="store_true", help="Run SofaScore tennis text-output mode.")
    parser.add_argument("--live_list", action="store_true", help="List current live tennis matches from SofaScore.")
    parser.add_argument("--live_event_id", type=int, default=None, help="SofaScore event id for the selected tennis match.")
    parser.add_argument("--live_query", default="", help="Search live matches by player, match, or tournament name.")
    parser.add_argument(
        "--live_detail",
        choices=["summary", "players", "sets", "stats", "all"],
        default=None,
        help="Text output type: summary, players, sets, stats, or all.",
    )
    parser.add_argument("--live_limit", type=int, default=20, help="Maximum live matches to show in lists.")
    parser.add_argument("--live_non_interactive", action="store_true", help="Use the first matching event without asking for input.")
    args = parser.parse_args()

    if args.live_info or args.live_list:
        try:
            run_live_info_mode(args)
        except KeyboardInterrupt:
            print("\nUser stopped the program.")
        except Exception as exc:
            print(f"Error: {exc}")
        return

    if cv2 is None or np is None:
        print("Video/hardware mode requires OpenCV and NumPy. Install cv2 and numpy, or run with --live_info.")
        return

    serial_conn = None
    if args.mode == "hardware" and args.serial_port:
        import serial
        serial_conn = serial.Serial(args.serial_port, 115200, timeout=1)
        time.sleep(2)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open {args.video}")
        return

    ret, first_frame = cap.read()
    if not ret: return

    court_points = select_court_points(first_frame)
    homography = build_homography(court_points)
    court_mask = make_court_roi_mask(first_frame.shape, court_points)

    detector = load_ball_detector(args.model_path)
    motion_detector = PureMotionBallDetector()
    player_suppressor = PlayerSuppressor(
        enabled=not args.disable_player_suppression,
        model_path=args.player_model_path,
        confidence=args.player_conf,
        stride=args.player_stride,
        margin=args.player_margin,
    )

    print("Starting processing...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Hybrid detection: YOLO + HSV + motion candidates are all considered.
        candidates = []
        candidates.extend(detect_yolo_candidates(frame, detector, args.conf, court_mask))
        candidates.extend(detect_hsv_candidates(frame, args.conf, court_mask))
        if not args.disable_streak_detector:
            candidates.extend(detect_streak_candidates(frame, args.conf, court_mask, args.streak_min_yellow))
        candidates.extend(motion_detector.detect(frame, args.conf, court_mask))
        round_rejected = 0
        if not args.disable_roundness_gate:
            candidates, round_rejected = filter_round_candidates(
                candidates,
                frame,
                args.roundness_threshold,
                args.yolo_roundness_threshold,
            )
        accessory_rejected = 0
        if not args.disable_accessory_filter:
            candidates, accessory_rejected = filter_accessory_candidates(
                candidates,
                args.generic_yolo_min_yellow,
                args.motion_min_yellow,
            )
        player_boxes = player_suppressor.update(frame, court_mask)
        candidates = filter_candidates_inside_players(candidates, player_boxes, args.player_overlap)

        best_candidate = max(candidates, key=score_candidate, default=None)

        if best_candidate:
            cx, cy = best_candidate.center
            # Homography map
            point = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, homography)[0, 0]
            tx, ty = transformed[0], transformed[1]

            # Grid calculation
            col = int(np.clip(tx // CELL_WIDTH, 0, GRID_COLS - 1))
            row = int(np.clip(ty // CELL_HEIGHT, 0, GRID_ROWS - 1))
            motor_id = row * GRID_COLS + col

            send_to_hardware(serial_conn, motor_id)
            cv2.circle(frame, (int(cx), int(cy)), 10, COLOR_GREEN, 2)
            draw_text(frame, f"Motor: {motor_id} ({best_candidate.source})", (20, 40))
        else:
            send_to_hardware(serial_conn, -1)

        for player_box in player_suppressor.boxes:
            x1, y1, x2, y2 = player_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
        if player_suppressor.boxes:
            draw_text(frame, f"player suppression: {len(player_suppressor.boxes)} box(es)", (20, 72), 0.6, COLOR_RED, 1)
        if not args.disable_roundness_gate:
            draw_text(frame, f"roundness rejected: {round_rejected}", (20, 102), 0.6, COLOR_BLUE, 1)
        if not args.disable_accessory_filter:
            draw_text(frame, f"accessory rejected: {accessory_rejected}", (20, 132), 0.6, COLOR_ORANGE, 1)

        cv2.imshow("Tennis Haptic Device - Filter Upgraded", resize_to_height(frame, 720))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if serial_conn:
        send_to_hardware(serial_conn, -1)
        serial_conn.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
