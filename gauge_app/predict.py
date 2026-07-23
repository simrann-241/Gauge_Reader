import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO
import logging
logger = logging.getLogger(__name__)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)
PRESET_ANGLE = 275

def calculate_angle(base, point):
    dx = point[0] - base[0]
    dy = point[1] - base[1]
    return (np.arctan2(dy, dx) * 180 / np.pi) % 360

class PressureGaugeReader:
    def __init__(self):
        self.center = None
        self.radius = None
        self.needle_line = None

    def detect_circle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, minDist=image.shape[0] * 0.5,
            param1=100, param2=50, minRadius=int(min(image.shape[:2]) * 0.1),
            maxRadius=int(min(image.shape[:2]) * 0.4)
        )
        if circles is not None:
            c = max(np.round(circles[0]).astype("int"), key=lambda c: c[2])
            return (c[0], c[1]), c[2]
        h, w = image.shape[:2]
        return (w//2, h//2), min(w, h)//3

    def detect_needle(self, image, center, radius):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        masked = cv2.bitwise_and(gray, mask)
        edges = cv2.Canny(masked, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(radius*0.5),
                                minLineLength=int(radius*0.4), maxLineGap=int(radius*0.1))
        if lines is None:
            return None
        best_score = float('inf')
        best_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            A, B = y2 - y1, x1 - x2
            if A == 0 and B == 0: continue
            C = x2*y1 - x1*y2
            dist = abs(A*center[0] + B*center[1] + C) / math.sqrt(A**2 + B**2)
            score = dist - np.linalg.norm([x2-x1, y2-y1]) * 0.1
            if score < best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)
        if best_line:
            x1, y1, x2, y2 = best_line
            dist1 = np.linalg.norm(np.array([x1, y1]) - np.array(center))
            dist2 = np.linalg.norm(np.array([x2, y2]) - np.array(center))
            if dist1 > dist2:
                tip, base = (x1, y1), (x2, y2)
            else:
                tip, base = (x2, y2), (x1, y1)
            return base, tip
        return None

    def estimate_min_max_positions(self, center, radius, start_angle=222.5, end_angle=-52.5):
        if end_angle < 0: end_angle += 360
        needle_len = radius * 0.8
        min_rad, max_rad = math.radians(start_angle), math.radians(end_angle)
        min_pos = (int(center[0] + needle_len * math.cos(min_rad)),
                   int(center[1] - needle_len * math.sin(min_rad)))
        max_pos = (int(center[0] + needle_len * math.cos(max_rad)),
                   int(center[1] - needle_len * math.sin(max_rad)))
        return min_pos, max_pos

    def fallback(self, image_path, min_value, max_value):
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to load image."
        center, radius = self.detect_circle(img)
        needle = self.detect_needle(img, center, radius)
        if not needle:
            return None, "Fallback needle detection failed."
        base, tip = needle
        min_pos, max_pos = self.estimate_min_max_positions(center, radius)
        return {
            "base_pos": base,
            "needle_tip": tip,
            "min_pos": min_pos,
            "max_pos": max_pos
        }, None

def predict_gauge(img_path, min_value, max_value):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, "Could not load image"

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_path)
        
        base_pos = needle_tip = min_pos = max_pos = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                x, y, *_ = box.xywh[0]
                px, py = x.item() * w, y.item() * h
                if cls == 0: base_pos = (px, py)
                elif cls == 3: needle_tip = (px, py)
                elif cls == 2: min_pos = (px, py)
                elif cls == 1: max_pos = (px, py)

        if not all([base_pos, needle_tip, min_pos, max_pos]):
            fallback = PressureGaugeReader()
            result, err = fallback.fallback(img_path, min_value, max_value)
            if err:
                return None, err
            base_pos = result["base_pos"]
            needle_tip = result["needle_tip"]
            min_pos = result["min_pos"]
            max_pos = result["max_pos"]

        min_angle = calculate_angle(base_pos, min_pos)
        max_angle = calculate_angle(base_pos, max_pos)
        needle_angle = calculate_angle(base_pos, needle_tip)

        rel_angle = (needle_angle - min_angle) % 360
        angle_range = PRESET_ANGLE
        rel_angle = max(0, min(rel_angle, angle_range))
        value = min_value + ((rel_angle / angle_range) * (max_value - min_value))

        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        ax.plot(*base_pos, 'go', label='Base')
        ax.plot(*needle_tip, 'bo', label='Needle Tip')
        ax.plot(*min_pos, 'ro', label='Min')
        ax.plot(*max_pos, 'yo', label='Max')
        ax.set_title(f"Gauge Reading: {value:.2f}")
        ax.axis('off')
        ax.legend()

        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        return round(value, 2), f"data:image/png;base64,{encoded}"
    
    except Exception as e:
        logger.exception("Prediction failed:")
        return None, "An unexpected error occurred. Check server logs."


