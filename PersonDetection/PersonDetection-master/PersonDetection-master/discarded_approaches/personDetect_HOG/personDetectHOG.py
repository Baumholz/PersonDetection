# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# actual processing algorithm
def processIMG(capture_device):
	cap = capture_device
	while (True):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		ret, image = cap.read()
		image = imutils.resize(image, width=min(400, image.shape[1]))
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		orig = image.copy()

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
												padding=(8, 8), scale=1.05)

		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# show the output images
		cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

time.sleep(0.1)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:

	# loading chipset driver (for raspberry pi)
	tmp = os.popen("ls -ltrh /dev/video*").read()
	if "/dev/video0" not in tmp:
		os.system("sudo modprobe bcm2835-v4l2")
		os.system("echo \"Camera Module has been loaded\"")
	else:
		tmp = os.system("echo \"Successfully initialized camera\"")

		# initialize the camera and grab a reference to the raw camera capture
		cap = cv2.VideoCapture(0)
		cap.set(4, 640)
		cap.set(5, 480)

		# allow the camera to warmup
		time.sleep(0.25)

		processIMG(cap)

		cap.release()
		cv2.destroyAllWindows()

# otherwise, we are reading from a video file
else:
	if os.path.isfile(cv2.VideoCapture(args["video"])):
		cap = cv2.VideoCapture(args["video"])
		processIMG(cap)
		cv2.destroyAllWindows()
	else:
		print("Fehler beim Zugriff auf Datei! Pfad überprüfen!")