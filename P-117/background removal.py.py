# import cv2 to capture videofeed
import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480
mountain = cv2.resize(mountain, (640, 480))

while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    # read a frame from the attached camera
    
    status, frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame, 1)
        img = np.flip(img, axis = 1)
        
        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([0, 120, 70])
        upper_bound = np.array([10, 255, 255])

        # thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        mask_inv = cv2.bitwise_not(mask)

        # bitwise and operation to extract foreground / person
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # final image
        final = cv2.bitwise_or(fg, mountain)

        # show it
        cv2.imshow('frame', final)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
cap.release()
out.release()
camera.release()
cv2.destroyAllWindows()