import cv2
import numpy as np
import mediapipe as mp
import os
import time

#config
brush_thickness = 15
eraser_thickness = 100
draw_color = (255, 0, 255) #purple
eraser_color = (0, 0, 0) #black to "erase"
smoothing_factor = 5

# folder
folder_path = "results"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

#canvas
#draw separate black canvas and merge it with the video
cap = cv2.VideoCapture(0)
#set camera resol
cap.set(3, 1280)
cap.set(4, 720)

img_canvas = None
xp, yp = 0, 0
undo_stack = []
tip_ids = [4, 8, 12, 16, 20] #thumb, index, middle, ring, pinky
# redo_stack = []

# UI colors
colors = [(255, 0, 255), #purple
          (255, 0, 0),   #blue
          (0, 255, 0),   #green
          (0, 0, 255),   #red
          (0, 0, 0)]     #black
header_height = 80

def draw_header(img, current_color):
    """Draws the color palette rectangles at the top of the screen"""
    # Draw a background bar
    cv2.rectangle(img, (0,0), (1280, header_height), (50, 50, 50), -1)
    
    # Draw Color Buttons (Width 1280 / 5 sections = ~250px each)
    # 1. Purple
    cv2.rectangle(img, (40, 10), (200, 70), (255, 0, 255), -1)
    if current_color == (255, 0, 255): 
        cv2.rectangle(img, (40, 10), (200, 70), (255, 255, 255), 3) # White border if selected

    # 2. Blue
    cv2.rectangle(img, (240, 10), (400, 70), (255, 0, 0), -1)
    if current_color == (255, 0, 0): 
        cv2.rectangle(img, (240, 10), (400, 70), (255, 255, 255), 3)

    # 3. Green
    cv2.rectangle(img, (440, 10), (600, 70), (0, 255, 0), -1)
    if current_color == (0, 255, 0): 
        cv2.rectangle(img, (440, 10), (600, 70), (255, 255, 255), 3)
        
    # 4. Red
    cv2.rectangle(img, (640, 10), (800, 70), (0, 0, 255), -1)
    if current_color == (0, 0, 255): 
        cv2.rectangle(img, (640, 10), (800, 70), (255, 255, 255), 3)

    # 5. Eraser (Clear) Button
    cv2.rectangle(img, (1050, 10), (1200, 70), (0, 0, 0), -1)
    cv2.putText(img, "CLEAR", (1080, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


print("Board Ready")

prev_time = 0

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    if img_canvas is None: img_canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw the Header Interface
    draw_header(img, draw_color)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []
            # Thumb
            if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]: fingers.append(1)
            else: fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]: fingers.append(1)
                else: fingers.append(0)

            x1, y1 = lm_list[8] # Index Tip

            # If hand is in the header area (y1 < 80)
            if y1 < header_height:
                if 40 < x1 < 200:
                    draw_color = (255, 0, 255) # Purple
                elif 240 < x1 < 400:
                    draw_color = (255, 0, 0) # Blue
                elif 440 < x1 < 600:
                    draw_color = (0, 255, 0) # Green
                elif 640 < x1 < 800:
                    draw_color = (0, 0, 255) # Red
                elif 1050 < x1 < 1200: # Clear Button
                    undo_stack.append(img_canvas.copy())
                    img_canvas = np.zeros((h, w, 3), np.uint8)
            
            # --- DRAWING LOGIC
            # Index up, Middle down, and NOT in header
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

            # --- ERASER LOGIC ---
            elif all(x == 1 for x in fingers):
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    undo_stack.append(img_canvas.copy())

                x_new = xp + (x1 - xp) // smoothing_factor
                y_new = yp + (y1 - yp) // smoothing_factor

                cv2.circle(img, (x1, y1), eraser_thickness // 2, eraser_color, cv2.FILLED)
                cv2.line(img_canvas, (xp, yp), (x_new, y_new), eraser_color, eraser_thickness)
                cv2.putText(img, "ERASING", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, eraser_color, 2)
                xp, yp = x_new, y_new

            else:
                xp, yp = 0, 0

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    else:
        xp, yp = 0, 0

    # Merge Canvas
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # --- FPS CALCULATION ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (1150, 680), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 2)

    # Instructions
    cv2.putText(img, "Index: Draw | Open Hand: Erase | u: Undo | s: Save", (10, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('u'):
        if undo_stack: img_canvas = undo_stack.pop()
    elif key == ord('s'):
        filename = f"results/drawing_{int(time.time())}.png"
        cv2.imwrite(filename, img_canvas)
        print(f"Saved: {filename}")

    cv2.imshow("Virtual Whiteboard", img)

cap.release()
cv2.destroyAllWindows()