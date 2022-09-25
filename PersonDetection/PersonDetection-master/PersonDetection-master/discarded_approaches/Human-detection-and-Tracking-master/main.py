# coding=utf-8
import argparse
import glob
import os
import time
import cv2
import imutils
from imutils.object_detection import non_max_suppression

font = cv2.FONT_HERSHEY_SIMPLEX                                                                     # font type
list_of_videos = []
hog = cv2.HOGDescriptor()                                                                           # creates the HOG descriptor and
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())                                    # detector with default params.
count = 0


def detect_people(frame):
    """
    detect humans using HOG descriptor
    The winStride  parameter is a 2-tuple that dictates the “step size” in both the x and y location of the sliding window
    The padding  parameter is a tuple which indicates the number of pixels in both the x and y direction
    in which the sliding window ROI is “padded” prior to HOG feature extraction.
    This scale  parameter controls the factor in which our image is resized at each layer of the image pyramid,
    ultimately influencing the number of levels in the image pyramid.
    """
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.2)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)                               # a float representing the threshold for deciding whether boxes overlap too much
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
    """
    This function returns 1 for the frames in which the area
    after subtraction with previous frame is greater than minimum area
    defined.

      First argument is the source image, which should be a grayscale image. Second argument is the threshold value which is used to classify the pixel values.
      Third argument is the maxVal which represents the value to be given if pixel value is more than (sometimes less than) the threshold value
    """
    frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)                               # calculates the per-element absolute difference between two arrays or between an array and a scalar.
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            temp = 1
    return temp


if __name__ == '__main__':
    """
    main function
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False,
                    help="path to videos directory")                                                 # gives the possibility to add a video
    args = vars(ap.parse_args())
    path = args["video"]
    if args.get("video", None) is None:                                                              # if no video path exist take the camera
                os.system("sudo modprobe bcm2835-v4l2")
                camera = cv2.VideoCapture(0)
                camera.set(3, 640)                                                                   # 3 is the width of the video stream
                camera.set(4, 480)                                                                   # 4 is the high
                camera.set(5, 20)                                                                    # 5 are the fps
                grabbed, frame = camera.read()                                                       # capture frame by frame
                frame_resized = imutils.resize(frame, width=min(640, frame.shape[1]))                # resized frame
                frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)            # Converts an image from one color space to another
                # print(frame_resized.shape)

                # defining min cuoff area
                min_area = (3000 / 800) * frame_resized.shape[1]

                while True:
                    starttime = time.time()                                                         # start time and endtime are needet to calculate the fps
                    previous_frame = frame_resized_grayscale
                    grabbed, frame = camera.read()
                    if not grabbed:
                        break
                    frame_resized = imutils.resize(frame, width=min(640, frame.shape[1]))
                    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    temp = background_subtraction(previous_frame, frame_resized_grayscale, min_area)# if there is an different between the 2 frames, temp = 1.
                    if temp == 1:                                                                   # That means something in the picture changed, and maybe there is a person
                        frame_processed = detect_people(frame_resized)                              # call the detect / HOG method
                        endtime = time.time()
                        cv2.putText(frame_processed, str(round(1 / (endtime - starttime))), (10, 10), font, 0.5,
                                    (0, 255, 0), 2)                                                 # shows the fps
                        cv2.imshow("Detected Human and face", frame_processed)                      # cv2.imshow("
                        key = cv2.waitKey(1) & 0xFF                                                 # display a frame for 1ms
                        if key == ord("q"):                                                         # when q gets pressed the person detection stops
                            break
                    # print("Time to process a frame: " + str(starttime - endtime))
                    else:
                        count = count + 1
                        print("Number of frame skipped in the video= " + str(count))

                camera.release()                                                                    # When everything done, release the capture
                cv2.destroyAllWindows()
    else:                                                                                           # When a video path is added
        for f in os.listdir(path):
            list_of_videos = glob.glob(os.path.join(os.path.abspath(path), f))                      # go to path
            for video in list_of_videos:
                camera = cv2.VideoCapture(os.path.join(path, video))                                # takes a video instead of a stream in the given location
                grabbed, frame = camera.read()
                # print(frame.shape)
                frame_resized = imutils.resize(frame, width=min(640, frame.shape[1]))
                frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                # print(frame_resized.shape)

                # defining min cuoff area
                min_area = (3000 / 800) * frame_resized.shape[1]

                while True:
                    starttime = time.time()
                    previous_frame = frame_resized_grayscale
                    grabbed, frame = camera.read()
                    if not grabbed:
                        break
                    frame_resized = imutils.resize(frame, width=min(640, frame.shape[1]))
                    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    temp = background_subtraction(previous_frame, frame_resized_grayscale, min_area)
                    if temp == 1:
                        frame_processed = detect_people(frame_resized)
                        endtime = time.time()
                        cv2.putText(frame_processed, str(round(1 / (endtime - starttime))), (10, 10), font, 0.5,
                                    (0, 255, 0), 2)
                        cv2.imshow("Detected Human and face", frame_processed)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    # print("Time to process a frame: " + str(starttime - endtime))
                    else:
                        count = count + 1
                        print("Number of frame skipped in the video= " + str(count))

                camera.release()
                cv2.destroyAllWindows()

else:
    print("model file not found")
list_of_videos = []
