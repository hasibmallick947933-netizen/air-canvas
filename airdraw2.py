"""
✋ Air Drawing with Hand Gestures MADE BY HASIB
==================================
Requirements:
    pip install opencv-python mediapipe numpy

Controls / Gestures:
    ✏️  INDEX finger only up      → DRAW mode
    ✋  ALL fingers up             → MOVE (no drawing)
    ✊  Fist (all fingers down)    → ERASE mode (eraser)
    🤏  Thumb + Index pinch        → CHANGE COLOR (cycles through palette)
    🤘  Index + Pinky up (horns)   → CLEAR canvas
    ✌️  Index + Middle up          → CHANGE brush SIZE
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ─────────────────────────────── CONFIG ─────────────────────────────── #
COLORS = [
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (130, 0, 75),   # Indigo
    (255, 0, 170),  # Violet
    (255, 255, 255),# White
]
COLOR_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet", "White"]
BRUSH_SIZES  = [3, 6, 10, 16, 24]
ERASER_SIZE  = 50
GESTURE_COOLDOWN = 0.6   # seconds between gesture triggers
# ─────────────────────────────────────────────────────────────────────── #


mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def fingers_up(landmarks, handedness="Right"):
    """Return list of 5 booleans: [Thumb, Index, Middle, Ring, Pinky]."""
    lm = landmarks.landmark
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]

    result = []
    # Thumb — compare x axis (mirrored for right hand)
    if handedness == "Right":
        result.append(lm[tips[0]].x < lm[joints[0]].x)
    else:
        result.append(lm[tips[0]].x > lm[joints[0]].x)
    # Other four fingers — compare y axis
    for i in range(1, 5):
        result.append(lm[tips[i]].y < lm[joints[i]].y)
    return result


def detect_gesture(up, landmarks):
    """Map finger state to a gesture string."""
    lm = landmarks.landmark

    # Fist
    if not any(up):
        return "ERASE"

    # All up → move
    if all(up):
        return "MOVE"

    # Index only → draw
    if up[1] and not up[2] and not up[3] and not up[4]:
        return "DRAW"

    # Index + Middle (✌️) → size
    if up[1] and up[2] and not up[3] and not up[4]:
        return "SIZE"

    # Index + Pinky (🤘) → clear
    if up[1] and not up[2] and not up[3] and up[4]:
        return "CLEAR"

    # Thumb + Index pinch (distance < threshold) → color
    if up[0] and up[1] and not up[2] and not up[3] and not up[4]:
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        dist = np.linalg.norm(thumb_tip - index_tip)
        if dist < 0.06:
            return "COLOR"

    return "NONE"


def put_label(img, text, pos, color=(255,255,255), scale=0.7, thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+4), (30,30,30), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_ui(frame, canvas, color, color_idx, brush_size, gesture, fps):
    h, w = frame.shape[:2]

    # Merge canvas onto frame
    output = frame.copy()
    mask   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    output[mask > 0] = canvas[mask > 0]

    # ── Top bar ──────────────────────────────────────────────────────── #
    overlay = output.copy()
    cv2.rectangle(overlay, (0,0), (w, 50), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    # Color swatches
    sw = 30
    for i, c in enumerate(COLORS):
        x0 = 10 + i * (sw+4)
        border = 3 if i == color_idx else 1
        cv2.rectangle(output, (x0-border, 8-border), (x0+sw+border, 8+sw+border), (200,200,200), border)
        cv2.rectangle(output, (x0,8), (x0+sw, 8+sw), c, -1)

    # Brush preview
    bx = 10 + len(COLORS)*(sw+4) + 20
    cv2.circle(output, (bx, 23), brush_size, color, -1)

    # Gesture indicator
    gesture_colors = {
        "DRAW":  (0,255,100),
        "ERASE": (0,100,255),
        "MOVE":  (200,200,200),
        "COLOR": (255,200,0),
        "SIZE":  (0,200,255),
        "CLEAR": (0,50,255),
        "NONE":  (120,120,120),
    }
    gc = gesture_colors.get(gesture, (255,255,255))
    icons = {"DRAW":"✏ DRAW","ERASE":"◎ ERASE","MOVE":"✋ MOVE",
             "COLOR":"🎨 COLOR","SIZE":"⊕ SIZE","CLEAR":"✗ CLEAR","NONE":"— —"}
    put_label(output, icons.get(gesture, gesture), (w-170, 35), gc, 0.65, 2, bg=False)

    # FPS
    put_label(output, f"FPS:{fps:3.0f}", (w-60, 35), (180,180,180), 0.5, 1, bg=False)

    # ── Legend ───────────────────────────────────────────────────────── #
    legend = [
        ("Index up",        "DRAW",  (0,255,100)),
        ("All fingers up",  "MOVE",  (200,200,200)),
        ("Fist",            "ERASE", (0,100,255)),
        ("Thumb+Index",     "COLOR", (255,200,0)),
        ("Index+Pinky",     "CLEAR", (0,50,255)),
        ("Index+Middle",    "SIZE",  (0,200,255)),
    ]
    lx, ly = 10, h - 10 - len(legend)*22
    for key, val, lc in legend:
        put_label(output, f"{key}: {val}", (lx, ly), lc, 0.45, 1)
        ly += 22

    return output


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        print("❌  Cannot open camera.")
        return

    h, w = first.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    color_idx  = 0
    brush_size = BRUSH_SIZES[1]
    brush_idx  = 1
    prev_x, prev_y = -1, -1
    gesture    = "NONE"
    last_triggered = 0.0
    fps_time   = time.time()
    fps        = 0

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    print("📷  Air Drawing started. Press 'q' or ESC to quit, 's' to save canvas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now = time.time()
        fps = 0.9*fps + 0.1*(1.0/(now - fps_time + 1e-9))
        fps_time = now

        if result.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                handedness = hand_info.classification[0].label  # "Left"/"Right"

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=1),
                )

                up = fingers_up(hand_lm, handedness)
                gesture = detect_gesture(up, hand_lm)

                # Index fingertip position
                ix = int(hand_lm.landmark[8].x * w)
                iy = int(hand_lm.landmark[8].y * h)

                color = COLORS[color_idx]

                # ── Gesture actions ────────────────────────────────── #
                if gesture == "DRAW":
                    if prev_x != -1:
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), color, brush_size)
                    prev_x, prev_y = ix, iy
                    cv2.circle(frame, (ix, iy), brush_size//2, color, -1)

                elif gesture == "ERASE":
                    cv2.circle(canvas, (ix, iy), ERASER_SIZE, (0,0,0), -1)
                    cv2.circle(frame, (ix, iy), ERASER_SIZE, (50,50,50), 2)
                    prev_x, prev_y = -1, -1

                elif gesture == "MOVE":
                    prev_x, prev_y = -1, -1

                elif gesture == "CLEAR" and (now - last_triggered > GESTURE_COOLDOWN):
                    canvas[:] = 0
                    last_triggered = now
                    prev_x, prev_y = -1, -1

                elif gesture == "COLOR" and (now - last_triggered > GESTURE_COOLDOWN):
                    color_idx = (color_idx + 1) % len(COLORS)
                    last_triggered = now
                    prev_x, prev_y = -1, -1

                elif gesture == "SIZE" and (now - last_triggered > GESTURE_COOLDOWN):
                    brush_idx = (brush_idx + 1) % len(BRUSH_SIZES)
                    brush_size = BRUSH_SIZES[brush_idx]
                    last_triggered = now
                    prev_x, prev_y = -1, -1

                else:
                    if gesture not in ("DRAW",):
                        prev_x, prev_y = -1, -1
        else:
            gesture = "NONE"
            prev_x, prev_y = -1, -1

        output = draw_ui(frame, canvas, COLORS[color_idx], color_idx, brush_size, gesture, fps)
        cv2.imshow("✋ Air Drawing", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break
        elif key == ord('s'):
            fname = f"air_drawing_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"💾  Saved canvas → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("👋  Bye!")


if __name__ == "__main__":
    main()