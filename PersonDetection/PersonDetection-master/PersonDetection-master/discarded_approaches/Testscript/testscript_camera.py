# import the necessary packages
import time
import cv2

#load chipset driver for camera module
import os
tmp = os.popen("ls -ltrh /dev/video*").read()
if "/dev/video0" not in tmp:
        os.system("sudo modprobe bcm2835-v4l2")
	os.system("echo \"Camera Module has been loaded\"")
else:
        tmp = os.system("echo \"Successfully initialized camera\"")

# initialize the camera and grab a reference to the raw camera capture
cap = cv2.VideoCapture(1)

# capture frames from the camera
while(True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	ret, image = cap.read()
	#image = cv2.flip(image,flipCode=-1)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
