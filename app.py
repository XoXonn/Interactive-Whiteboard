import cv2
import numpy as np
import mediapipe as mp
import os
import time

# --- CONFIGURATION ---
draw_color = (255, 0, 255) 
brush_thickness = 15
eraser_thickness = 100 
smoothing_factor = 5       
background_blur_on = False # Start with blur off

# --- FOLDER SETUP ---
if not os.path.exists("results"): os.makedirs("results")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # 0=General, 1=Landscape(faster)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

img_canvas = None
xp, yp = 0, 0
undo_stack = []
tip_ids = [4, 8, 12, 16, 20] 

# --- UI DEFINITIONS ---
ui_elements = [
    # COLORS
    {"x": 30,  "y": 10, "w": 70, "h": 70, "color": (255, 0, 255), "type": "color", "value": (255, 0, 255)}, 
    {"x": 120, "y": 10, "w": 70, "h": 70, "color": (255, 0, 0),   "type": "color", "value": (255, 0, 0)},   
    {"x": 210, "y": 10, "w": 70, "h": 70, "color": (0, 255, 0),   "type": "color", "value": (0, 255, 0)},   
    {"x": 300, "y": 10, "w": 70, "h": 70, "color": (0, 0, 255),   "type": "color", "value": (0, 0, 255)},   
    {"x": 390, "y": 10, "w": 70, "h": 70, "color": (0, 255, 255), "type": "color", "value": (0, 255, 255)}, 

    # SIZES
    {"x": 520, "y": 10, "w": 70, "h": 70, "color": (50, 50, 50), "type": "size", "value": 5, "label": "S"},
    {"x": 610, "y": 10, "w": 70, "h": 70, "color": (50, 50, 50), "type": "size", "value": 15, "label": "M"},
    {"x": 700, "y": 10, "w": 70, "h": 70, "color": (50, 50, 50), "type": "size", "value": 30, "label": "L"},

    # ACTIONS
    {"x": 850, "y": 20, "w": 90, "h": 50, "color": (50, 50, 50), "type": "btn", "action": "undo", "label": "UNDO"},
    {"x": 950, "y": 20, "w": 90, "h": 50, "color": (50, 50, 50), "type": "btn", "action": "clear", "label": "CLEAR"},
    {"x": 1050, "y": 20, "w": 90, "h": 50, "color": (50, 50, 50), "type": "btn", "action": "save", "label": "SAVE"},
    
    # BLUR TOGGLE (New)
    {"x": 1150, "y": 20, "w": 90, "h": 50, "color": (50, 50, 50), "type": "toggle", "action": "blur", "label": "BLUR"},
]

def draw_ui(img, curr_color, curr_thickness, blur_active):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 90), (30, 30, 30), -1)
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for btn in ui_elements:
        x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']
        cx, cy = x + w // 2, y + h // 2

        if btn['type'] == 'color':
            cv2.circle(img, (cx, cy), 30, btn['color'], -1)
            if curr_color == btn['value']:
                cv2.circle(img, (cx, cy), 34, (255, 255, 255), 3)

        elif btn['type'] == 'size':
            color = (80, 80, 80)
            if curr_thickness == btn['value']: color = (150, 150, 150)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            radius = btn['value'] // 2 + 5
            cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1)

        elif btn['type'] == 'btn':
            cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 70), -1)
            cv2.putText(img, btn['label'], (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif btn['type'] == 'toggle':
            # Green if active, Gray if inactive
            bg_color = (0, 150, 0) if blur_active else (70, 70, 70)
            cv2.rectangle(img, (x, y), (x+w, y+h), bg_color, -1)
            cv2.putText(img, btn['label'], (x+20, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

print("Whiteboard Ultimate Running...")
prev_time = 0

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    if img_canvas is None: img_canvas = np.zeros((h, w, 3), np.uint8)
    
    # --- 1. BACKGROUND BLUR LOGIC ---
    if background_blur_on:
        # Convert to RGB for MediaPipe
        img_rgb_seg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_seg = segmentation.process(img_rgb_seg)
        
        # Create a condition mask (True where person is, False where background is)
        condition = np.stack((results_seg.segmentation_mask,) * 3, axis=-1) > 0.1
        
        # Create Blurred Background
        bg_image = cv2.GaussianBlur(img, (55, 55), 0)
        
        # Combine: Where condition is True (Person), use img; else use bg_image
        img = np.where(condition, img, bg_image)

    # --- 2. HAND TRACKING ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []
            if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]: fingers.append(1)
            else: fingers.append(0)
            for id in range(1, 5):
                if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]: fingers.append(1)
                else: fingers.append(0)

            x1, y1 = lm_list[8] 

            # UI INTERACTION
            if y1 < 90:
                for btn in ui_elements:
                    if btn['x'] < x1 < btn['x'] + btn['w']:
                        if btn['type'] == 'color': draw_color = btn['value']
                        elif btn['type'] == 'size': brush_thickness = btn['value']
                        elif btn['type'] == 'btn':
                            if btn['action'] == 'clear':
                                undo_stack.append(img_canvas.copy())
                                img_canvas = np.zeros((h, w, 3), np.uint8)
                            elif btn['action'] == 'undo':
                                if undo_stack: 
                                    img_canvas = undo_stack.pop()
                                    time.sleep(0.2)
                            elif btn['action'] == 'save':
                                cv2.imwrite(f"results/drawing_{int(time.time())}.png", img_canvas)
                                cv2.putText(img, "SAVED!", (x1, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                time.sleep(0.2)
                        elif btn['type'] == 'toggle':
                            if btn['action'] == 'blur':
                                background_blur_on = not background_blur_on
                                time.sleep(0.2) # Debounce

            # DRAWING
            elif fingers[1] == 1 and fingers[2] == 0:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    if len(undo_stack) > 10: undo_stack.pop(0)
                    undo_stack.append(img_canvas.copy())

                x_new = xp + (x1 - xp) // smoothing_factor
                y_new = yp + (y1 - yp) // smoothing_factor
                cv2.circle(img, (x1, y1), brush_thickness // 2, draw_color, cv2.FILLED)
                cv2.line(img_canvas, (xp, yp), (x_new, y_new), draw_color, brush_thickness)
                xp, yp = x_new, y_new

            # ERASER
            elif all(x == 1 for x in fingers):
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    undo_stack.append(img_canvas.copy())
                x_new = xp + (x1 - xp) // smoothing_factor
                y_new = yp + (y1 - yp) // smoothing_factor
                cv2.circle(img, (x1, y1), eraser_thickness // 2, (0,0,0), cv2.FILLED)
                cv2.line(img_canvas, (xp, yp), (x_new, y_new), (0,0,0), eraser_thickness)
                cv2.putText(img, "ERASING", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                xp, yp = x_new, y_new

            else:
                xp, yp = 0, 0
            
            # Draw landmarks AFTER blur so they are sharp
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    else:
        xp, yp = 0, 0

    img = draw_ui(img, draw_color, brush_thickness, background_blur_on)

    # MERGE CANVAS
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (1180, 700), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Virtual Whiteboard Ultimate", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()