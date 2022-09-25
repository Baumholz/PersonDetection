# This program tries to recognize people crossing an entrance area and count them by means of a tracking algorithm.
# The countings are sent via the LoRa protocol.
import datetime
import os
import argparse
import cv2
import math

class Track():
    def __init__(self):
        self.spur = []  # type: List[Person]
        self.timesTracked = 0
        self.isTracked = False

class Person():
    def __init__(self):
        self.xCentroid = 0
        self.yCentroid = 0
        self.width = 0
        self.height = 0
        self.direction = 0

counterTotal = 0
counterEntrance = 0
counterExit = 0
tsLastWrite = datetime.datetime.utcnow()
tracks = []
newRecognitions = []
width = 640
height = 480

# send counter values via lora in defined interval
def lora(seconds):
    global tsLastWrite
    global counterEntrance
    global counterExit
    now = datetime.datetime.utcnow()
    if (now - tsLastWrite).seconds >= seconds:
        print("Trying to send data to ttn ...")
        # shell command with path to the LoRa sending script, with counter values as parameter
        os.system("sudo ~/Documents/GitRepo_PedestrianDetection/LoRa/ttnsender/main/geotagger " + str(counterEntrance) + " " + str(
            counterExit) + " &")
        tsLastWrite = now
        counterEntrance = 0
        counterExit = 0

def draw(originalFrame, mask):
    ##originalFrame = imutils.resize(originalFrame, 640, 480)
    cv2.putText(originalFrame, "Counter: {}".format(str(counterTotal)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.putText(originalFrame, "Entrance: {}".format(str(counterEntrance)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.putText(originalFrame, "Exit: {}".format(str(counterExit)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.imshow('original', originalFrame)
    cv2.moveWindow('original', 200, 200)
    #mask = imutils.resize(mask, 640, 480)
    cv2.imshow('bs', mask)
    cv2.moveWindow('bs', 850, 200)

def drawTracks():
    for t in tracks:
        if len(t.spur)>1:
            for i in range(len(t.spur)-1):
                cv2.line(frame, (t.spur[i].xCentroid,t.spur[i].yCentroid),(t.spur[i+1].xCentroid,t.spur[i+1].yCentroid),(0,255,0),1)

def backgroundSubstraction(frame, backgroundSub):
    bsMask = backgroundSub.apply(frame)
    bsMask = cv2.morphologyEx(bsMask, cv2.MORPH_CLOSE, (20, 20))
    bsMask = cv2.dilate(bsMask, (10, 10), iterations=1)
    # Gaussion Blur for larger Blobs and smoother edges
    bsMask = cv2.GaussianBlur(bsMask, (21, 21), 2)
    bsMask = cv2.medianBlur(bsMask, 5)
    return bsMask

# copyOfCounts = 0
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the frame from the camera
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# create Background Substraction Object without shadow detection
backgroundSub = cv2.createBackgroundSubtractorMOG2(0, 100, False)

# image processing
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # uncomment for better performance while debugging
    #if grabbed:
     #   frame = imutils.resize(frame, 320)

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # applying Background Substraction Algorithm
    bsMask = backgroundSubstraction(frame, backgroundSub)
    # finding blobs in Backround Substraction mask
    _, cnts, _ = cv2.findContours(bsMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # minimum area of a blob dependend on frame size
    minCountourArea = 7000 * (width * height / (640 * 480))

    # processing for every recognized blob
    for c in cnts:
        # if a contour has small area, we assume it is not a person
        if cv2.contourArea(c) < minCountourArea:
            continue

        # draw an rectangle "around" the person
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(cv2.contourArea(c)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_4)
        # a Person is recognized
        rec = Person()
        # find blob centroid
        rec.xCentroid = int(round((x + x + w) / 2))
        rec.yCentroid = int(round((y + y + h) / 2))
        ObjectCentroid = (rec.xCentroid, rec.yCentroid)
        # draw Centroid
        cv2.circle(frame, ObjectCentroid, 1, (255, 0, 100), 5)
        newRecognitions.append(rec)

    #
    for person in newRecognitions:
        closest = None
        trackedOnce = False
        if len(tracks) > 0:
            closest = tracks[0]
        for t in tracks:
            dist = math.sqrt((t.spur[-1].xCentroid - person.xCentroid) ** 2 + (t.spur[-1].yCentroid - person.yCentroid) ** 2)
            distClosest = math.sqrt(
                (closest.spur[-1].xCentroid - person.xCentroid) ** 2 + (closest.spur[-1].yCentroid - person.yCentroid) ** 2)
            # find next blob to a track when distance is close enough
            if dist < 100 * (width/640):
                if dist <= distClosest:
                    closest = t
                    # timesTracked correspond to the number of frames until the track is deleted an counter is incremented
                    closest.timesTracked = 10
                    t.isTracked = True
                    trackedOnce = True
        # add the person to the closest track when one exists
        if trackedOnce:
            closest.spur.append(person)
            trackedOnce = False
        # could not assign blob to an existing Track
        else:
            temp = Track()
            #del temp.spur[:]
            temp.spur.append(person)
            tracks.append(temp)

    # old tracks are deleted when there are no new recognitions assigned to them
    for t in tracks:
        # every time there is no assignment to a track the times tracked counter is decremented
        # so that we can detect people moving out of camera range
        if t.isTracked is False:
            t.timesTracked -= 1
        else:
            t.isTracked = False
        if t.timesTracked is 0:
            # increment counter when tracking distance between first and last blob is great enough
            dist = t.spur[0].yCentroid - t.spur[-1].yCentroid
            if abs(dist) >= (height*0.3):
                counterTotal += 1
                if dist < 0:
                    counterEntrance += 1
                else:
                    counterExit += 1
            tracks.remove(t)


    drawTracks()
    del newRecognitions[:]
    draw(frame, bsMask)
    lora(60)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
print('Total: ' + str(counterTotal))
print('Entrance: ' + str(counterEntrance))
print('Exit: ' + str(counterExit))
camera.release()
cv2.destroyAllWindows()
