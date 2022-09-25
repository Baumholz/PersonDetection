import time
import numpy as np
import cv2
 
# initialize the camera and grab a reference to the raw camera capture
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))

#load chipset driver for camera module
import os
tmp = os.popen("ls -ltrh /dev/video*").read()
if "/dev/video0" not in tmp:
        os.system("sudo modprobe bcm2835-v4l2")
        os.system("echo \"Camera Module has been loaded\"")
else:
        tmp = os.system("echo \"Successfully initialized camera\"")


cap = cv2.VideoCapture(0)
cap.set(4,320)
cap.set(5,240)

#create Backgroundsubstractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()
 
# allow the camera to warmup
#time.sleep(0.1)
 
# capture frames from the camera
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while(True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	ret, image = cap.read()
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	fgmask = fgbg.apply(image)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	# show the frame
	cv2.imshow("Frame", fgmask)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	#rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
