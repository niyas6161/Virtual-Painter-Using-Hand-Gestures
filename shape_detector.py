import cv2
import numpy as np
import HandTrackingModule as htm

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # Detect shapes
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif len(approx) == 6:
            shape = "Hexagon"
        else:
            # Check if contour is a circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity >= 0.85:
                shape = "Circle"
            else:
                shape = ""

        detected_shapes.append((shape, (x, y, w, h)))

    return detected_shapes


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.8)

drawingColor = (0, 0, 255)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
overlay = np.ones_like(imgCanvas) * 255  # White canvas for overlay

eraserSize = 50
minBrushSize = 1
maxBrushSize = 50

xp, yp = 0, 0

# Stack to store history for undo and redo
history = []
redo_stack = []

while True:
    # 1. Preprocess screen
    success, image = cap.read()
    image = cv2.flip(image, 1)

    # 2. Overlay rectangles and text
    overlay = np.ones_like(imgCanvas) * 255  # Reset overlay
    cv2.rectangle(overlay, (0, 0), (1280, 110), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(overlay, (10, 10), (230, 50), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(overlay, (250, 10), (470, 50), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(overlay, (490, 10), (710, 50), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(overlay, (730, 10), (950, 50), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(overlay, (970, 10), (1270, 50), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(overlay, (10, 60), (130, 100), (255, 0, 255), cv2.FILLED)
    cv2.rectangle(overlay, (1150, 60), (1270, 100), (255, 255, 0), cv2.FILLED)
    cv2.putText(overlay, 'Eraser', (1070, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(overlay, 'Undo', (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(overlay, 'Redo', (1170, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    # 3. Find hand landmarks
    image_with_landmarks = image.copy()
    image_with_landmarks = detector.findHands(image_with_landmarks)
    lmlist = detector.findPosition(image_with_landmarks)
    # Combine canvas with overlay
    image = cv2.addWeighted(image, 0.5, overlay, 1, 0)

    # Copy the current canvas state
    current_canvas = imgCanvas.copy()

    x1,y1=0,0
    

    # Calculate distance between finger[8] and finger[4]
    if len(lmlist) >= 9:
        x0, y0 = lmlist[8][1:]  # Index finger
        x1, y1 = lmlist[4][1:]  # Thumb
        # Calculate Euclidean distance
        distance = int(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
        # Adjust brush size based on distance
        brushSize = np.interp(distance, [50, 250], [minBrushSize, maxBrushSize])  # Interpolate brush size based on distance

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # Finger 1
        x2, y2 = lmlist[12][1:]  # Finger 2

        # 4. Check which finger is up
        fingers = detector.fingersUp()

        # 5. Selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            # Color selection
            if y1 < 60:
                if 10 < x1 < 230:
                    drawingColor = (0, 0, 255)
                elif 250 < x1 < 470:
                    drawingColor = (0, 255, 0)
                elif 490 < x1 < 710:
                    drawingColor = (255, 0, 0)
                elif 730 < x1 < 950:
                    drawingColor = (0, 255, 255)
                elif 970 < x1 < 1270:
                    drawingColor = (0, 0, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), drawingColor, cv2.FILLED)

        # 6. Drawing mode - one finger is up
        if fingers[1] and not fingers[2] and y1 > 110:  # Only allow drawing below y-coordinate 110
            # Ensure drawing coordinates are within the canvas bounds
            x1 = min(max(x1, 0), 1280)
            y1 = min(max(y1, 110), 720)

            cv2.circle(image, (x1, y1), int(brushSize), drawingColor, thickness=-1)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawingColor == (0, 0, 0):
                cv2.line(image, (xp, yp), (x1, y1), drawingColor, eraserSize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawingColor, eraserSize)
            else:
                cv2.line(image, (xp, yp), (x1, y1), drawingColor, int(brushSize))
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawingColor, int(brushSize))

            xp, yp = x1, y1

     # Detect shapes on the canvas
    detected_shapes = detect_shapes(imgCanvas)
    for shape, (x, y, w, h) in detected_shapes:
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.drawContours(image, [np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])], 0, (0, 0, 255), 2)        

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)


    cv2.imshow('virtual painter', image)
    # Resize and display the window with hand landmarks
    if image_with_landmarks is not None:
        cv2.imshow('Hand Landmarks', cv2.resize(image_with_landmarks, (400, 300)))  # Resize to 400x300
    
    # Handle undo and redo functionality
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    
    elif y1 < 120:
        if 10 < x1 < 130:  # Undo
            if len(history) > 0:
                redo_stack.append(imgCanvas.copy())
                imgCanvas = history.pop()
        elif 1150 < x1 < 1270:  # Redo
            if len(redo_stack) > 0:
                history.append(imgCanvas.copy())
                imgCanvas = redo_stack.pop()
    else:
        # Store current canvas state in history
        if not np.array_equal(current_canvas, imgCanvas):
            history.append(current_canvas.copy())

cap.release()
cv2.destroyAllWindows()
