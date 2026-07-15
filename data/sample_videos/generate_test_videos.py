"""
Generate synthetic surveillance-style test videos for person detection testing.
Each video simulates a different CCTV scene: office, warehouse, corridor, parking, shopping_mall.
People are represented as upright silhouettes (rectangles with oval heads) moving through the frame.
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FPS = 15
DURATION_SEC = 10  # 10 seconds each


def draw_person(frame, x, y, scale=1.0, color=(60, 60, 60)):
    """Draw a simple person silhouette (body + head) at (x, y) center-bottom."""
    body_w = int(24 * scale)
    body_h = int(50 * scale)
    head_r = int(11 * scale)

    # Body
    cv2.rectangle(frame,
                  (x - body_w // 2, y - body_h),
                  (x + body_w // 2, y),
                  color, -1)
    # Head
    cv2.circle(frame, (x, y - body_h - head_r), head_r, color, -1)


def add_noise(frame, sigma=8):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_timestamp(frame, frame_idx, fps, scene):
    import datetime
    secs = frame_idx / fps
    ts = datetime.datetime(2024, 6, 15, 8, 0, 0) + datetime.timedelta(seconds=secs)
    text = f"{scene.upper()}  {ts.strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)


def make_video(filename, width, height, bg_fn, people_fn, scene_name):
    path = os.path.join(OUTPUT_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, FPS, (width, height))
    total_frames = FPS * DURATION_SEC
    for i in range(total_frames):
        frame = bg_fn(width, height, i, total_frames)
        people_fn(frame, i, total_frames)
        add_timestamp(frame, i, FPS, scene_name)
        frame = add_noise(frame, sigma=5)
        out.write(frame)
    out.release()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB, {width}x{height} @ {FPS}fps, {DURATION_SEC}s")
    return path


# ─── Scene 1: office ────────────────────────────────────────────────────────
def office_bg(w, h, i, n):
    frame = np.full((h, w, 3), (190, 185, 175), dtype=np.uint8)
    # Floor line
    cv2.line(frame, (0, int(h * 0.72)), (w, int(h * 0.72)), (130, 125, 115), 2)
    # Two desk shapes
    cv2.rectangle(frame, (20, int(h * 0.55)), (160, int(h * 0.72)), (100, 80, 60), -1)
    cv2.rectangle(frame, (w - 170, int(h * 0.55)), (w - 20, int(h * 0.72)), (100, 80, 60), -1)
    # Window on wall
    cv2.rectangle(frame, (w // 2 - 60, 30), (w // 2 + 60, 120), (200, 220, 240), -1)
    cv2.rectangle(frame, (w // 2 - 60, 30), (w // 2 + 60, 120), (140, 140, 140), 2)
    return frame


def office_people(frame, i, n):
    h, w = frame.shape[:2]
    t = i / n
    # Person 1: walks left to right
    x1 = int(80 + t * (w - 160))
    draw_person(frame, x1, int(h * 0.72), scale=1.0, color=(50, 50, 80))
    # Person 2: stationary at desk
    draw_person(frame, w - 100, int(h * 0.72), scale=0.9, color=(70, 50, 50))
    # Person 3: enters from left halfway through
    if t > 0.5:
        x3 = int((t - 0.5) / 0.5 * 120)
        draw_person(frame, x3, int(h * 0.72), scale=0.85, color=(40, 70, 40))


# ─── Scene 2: warehouse ─────────────────────────────────────────────────────
def warehouse_bg(w, h, i, n):
    frame = np.full((h, w, 3), (80, 75, 70), dtype=np.uint8)
    # Shelving units
    for sx in [30, 200, 380, 560]:
        if sx + 100 < w:
            cv2.rectangle(frame, (sx, int(h * 0.15)), (sx + 90, int(h * 0.75)), (55, 50, 45), -1)
            for shelf_y in [0.25, 0.45, 0.60]:
                cv2.rectangle(frame, (sx, int(h * shelf_y)), (sx + 90, int(h * shelf_y) + 6), (40, 35, 30), -1)
    # Floor
    cv2.rectangle(frame, (0, int(h * 0.75)), (w, h), (65, 60, 55), -1)
    # Overhead light patches
    for lx in range(0, w, 120):
        cv2.circle(frame, (lx, 0), 60, (100, 95, 85), -1)
    return frame


def warehouse_people(frame, i, n):
    h, w = frame.shape[:2]
    t = i / n
    floor_y = int(h * 0.75)
    # Forklift operator walks aisle
    x1 = int(140 + t * 250)
    draw_person(frame, x1, floor_y, scale=1.1, color=(30, 60, 100))
    # Worker going opposite direction
    x2 = int(w - 80 - t * 200)
    draw_person(frame, x2, floor_y, scale=1.0, color=(100, 60, 30))


# ─── Scene 3: corridor ───────────────────────────────────────────────────────
def corridor_bg(w, h, i, n):
    frame = np.full((h, w, 3), (170, 165, 155), dtype=np.uint8)
    # Perspective corridor walls
    vp_x, vp_y = w // 2, h // 3
    for side in [-1, 1]:
        pts = np.array([
            [w // 2 + side * w // 2, 0],
            [vp_x + side * 15, vp_y],
            [vp_x + side * 15, h],
            [w // 2 + side * w // 2, h],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (145, 140, 130))
    # Floor
    floor_pts = np.array([[0, h], [vp_x - 15, vp_y], [vp_x + 15, vp_y], [w, h]], dtype=np.int32)
    cv2.fillPoly(frame, [floor_pts], (120, 115, 105))
    # Ceiling light strip
    cv2.line(frame, (vp_x - 5, vp_y), (vp_x, 0), (230, 225, 210), 3)
    cv2.line(frame, (vp_x + 5, vp_y), (vp_x, 0), (230, 225, 210), 3)
    # Doors on walls
    for door_t in [0.3, 0.6]:
        dx = int(vp_x - 15 + (vp_x * 0.5) * (1 - door_t))
        dw = int(18 * door_t)
        dh = int(55 * door_t)
        dy = int(vp_y + (h - vp_y) * door_t - dh)
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (100, 95, 85), -1)
    return frame


def corridor_people(frame, i, n):
    h, w = frame.shape[:2]
    t = i / n
    vp_y = h // 3
    # Person approaching camera (grows over time)
    scale1 = 0.3 + t * 0.8
    x1 = w // 2 + int((t - 0.5) * 60)
    y1 = int(vp_y + (h - vp_y) * (0.2 + t * 0.7))
    draw_person(frame, x1, y1, scale=scale1, color=(40, 40, 60))
    # Second person receding
    if t < 0.7:
        scale2 = 0.9 - t * 0.6
        y2 = int(vp_y + (h - vp_y) * (0.9 - t * 0.7))
        draw_person(frame, w // 2 - 20, y2, scale=scale2, color=(60, 40, 40))


# ─── Scene 4: parking ────────────────────────────────────────────────────────
def parking_bg(w, h, i, n):
    # Top-down / slight overhead view
    frame = np.full((h, w, 3), (90, 88, 82), dtype=np.uint8)
    # Parking lines
    for col in range(0, w, 80):
        cv2.line(frame, (col, int(h * 0.1)), (col, int(h * 0.9)), (130, 128, 122), 1)
    # Two parked cars
    cv2.rectangle(frame, (40, int(h * 0.2)), (110, int(h * 0.5)), (50, 60, 80), -1)
    cv2.rectangle(frame, (200, int(h * 0.2)), (270, int(h * 0.5)), (80, 50, 50), -1)
    # Walkway marking
    for stripe in range(int(h * 0.6), int(h * 0.8), 20):
        cv2.line(frame, (w // 3, stripe), (2 * w // 3, stripe), (200, 195, 180), 2)
    return frame


def parking_people(frame, i, n):
    h, w = frame.shape[:2]
    t = i / n
    # Person walks through parking lot
    x1 = int(w * 0.15 + t * w * 0.6)
    y1 = int(h * 0.70)
    draw_person(frame, x1, y1, scale=0.9, color=(30, 50, 80))
    # Person near cars (appears mid-video)
    if t > 0.3:
        t2 = (t - 0.3) / 0.7
        x2 = int(140 + t2 * 60)
        draw_person(frame, x2, int(h * 0.55), scale=0.85, color=(70, 40, 40))


# ─── Scene 5: shopping_mall ──────────────────────────────────────────────────
def mall_bg(w, h, i, n):
    frame = np.full((h, w, 3), (210, 205, 195), dtype=np.uint8)
    # Tiled floor
    tile = 50
    for tx in range(0, w, tile):
        cv2.line(frame, (tx, int(h * 0.6)), (tx, h), (185, 180, 170), 1)
    for ty in range(int(h * 0.6), h, tile // 2):
        cv2.line(frame, (0, ty), (w, ty), (185, 180, 170), 1)
    # Shop fronts
    shop_colors = [(160, 130, 110), (110, 140, 160), (130, 160, 110), (155, 120, 150)]
    shop_w = w // 4
    for idx, sc in enumerate(shop_colors):
        sx = idx * shop_w
        cv2.rectangle(frame, (sx, 0), (sx + shop_w - 4, int(h * 0.6)), sc, -1)
        # Shop sign
        cv2.rectangle(frame, (sx + 10, 10), (sx + shop_w - 14, 45), (240, 235, 225), -1)
    # Ceiling lights
    for lx in range(60, w, 120):
        cv2.rectangle(frame, (lx - 20, 0), (lx + 20, 5), (245, 240, 230), -1)
    return frame


def mall_people(frame, i, n):
    h, w = frame.shape[:2]
    t = i / n
    floor_y = int(h * 0.95)
    configs = [
        (0.05, 0.7, 1.0, (40, 40, 70)),    # left to right, full video
        (0.0, -0.5, 0.9, (70, 40, 40)),     # right to left
        (0.25, 0.3, 0.85, (40, 70, 40)),    # slower, mid-frame
        (0.5, 0.4, 0.8, (60, 60, 40)),      # enters late
    ]
    for start_t, speed, scale, color in configs:
        if t >= start_t:
            local_t = (t - start_t) / max(1 - start_t, 0.01)
            if speed > 0:
                x = int(w * 0.05 + local_t * speed * w)
            else:
                x = int(w * 0.95 + local_t * speed * w)
            x = max(20, min(w - 20, x))
            draw_person(frame, x, floor_y, scale=scale, color=color)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    scenes = [
        ("office.mp4",        640, 360, office_bg,    office_people,    "OFFICE CAM-01"),
        ("warehouse.mp4",     640, 360, warehouse_bg,  warehouse_people, "WAREHOUSE CAM-02"),
        ("corridor.mp4",      640, 360, corridor_bg,   corridor_people,  "CORRIDOR CAM-03"),
        ("parking.mp4",       640, 360, parking_bg,    parking_people,   "PARKING CAM-04"),
        ("shopping_mall.mp4", 640, 360, mall_bg,       mall_people,      "MALL CAM-05"),
    ]

    print(f"Generating {len(scenes)} synthetic surveillance test videos...")
    print(f"Output dir: {OUTPUT_DIR}\n")

    for filename, w, h, bg_fn, ppl_fn, scene_name in scenes:
        make_video(filename, w, h, bg_fn, ppl_fn, scene_name)

    print("\nVerifying videos with OpenCV...")
    all_ok = True
    for filename, w, h, *_ in scenes:
        path = os.path.join(OUTPUT_DIR, filename)
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        status = "OK" if ok and frames > 0 else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {filename}: {vw}x{vh} @ {fps}fps, {frames} frames")

    print("\nAll videos ready." if all_ok else "\nSome videos failed verification.")


if __name__ == "__main__":
    main()
