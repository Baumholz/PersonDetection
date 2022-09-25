############################################################################################################
############ This implementation is based on a Project by Adrian Rosebrock from June 1, 2015 ###############
#Link to the Project:
# https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/
############################################################################################################

# import the necessary packages
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import os
import math
import numpy as np

# stores the information, where the recognized blob "shape" has been tracked (array stores coordinates of track)
class Person:
    def __init__(self, shape, counted=False):
        self.shape = shape
        self.counted = counted
        self.pos = None       # stores top left corner for persons going down and bottom left corner for persons going up
        self.ts = datetime.datetime.utcnow().second     # stores timestamp, how since when the last actualisation has been
        self.movementVector = (0, 0)
        self.oben = False

# create counter for persons in total
personCounterRein = 0
personCounterRaus = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
ap.add_argument("-v", "--video", required=False, help="path to the video file")
args = vars(ap.parse_args())

# filter warnings and load the configuration client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# init list for recognized persons
tracked_pers = []

#variables for debugging
stepByStep = False
nextframe = False

#stores the timestamp since the last lora-package has been sent
tsLastWrite = datetime.datetime.utcnow()

# calls the lora script periodically, passes person-counts as parameters and resets them afterwards
def lora():
    global tsLastWrite
    global personCounterRein
    global personCounterRaus
    now = datetime.datetime.utcnow()
    if ((now - tsLastWrite).seconds >= conf["sec_between_send"]):
        print("Trying to send data to ttn ...")
        os.system("sudo " + conf["path_to_geotagger"] + " " + str(personCounterRein) + " " + str(
            personCounterRaus) + " &")

        tsLastWrite = now
        personCounterRein = 0
        personCounterRaus = 0

# removes person from array if recognition box around person touches the frame border
# returns True, if Person has been removed
def removeFromTracking(xp, yp, wp, hp, idxp):
    if (yp <= 0 or yp + hp >= conf["resolution"][1]):
        if (conf["debugging"]):
            print("removing person")
        del tracked_pers[idxp]
        return True
    else:
        return False

# draws a rectangular frame(green) around tracked person
def drawRecFrame(frame, xp, yp, wp, hp, p, idxp):
    xcenter = xp + int(round(0.5 * wp))
    ycenter = yp + int(round(0.5 * hp))
    cv2.rectangle(frame, (xp, yp), (xp + wp, yp + hp), (0, 255, 0), 2)
    cv2.circle(frame, (xcenter, ycenter), 2, (0, 255, 0), 2)  # rect center
    cv2.putText(frame, str(idxp), (xp + 10, yp + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, str(cv2.contourArea(p.shape)), (xp + 10, yp + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

# checks if track is possible, otherwise estimates the correct position and adds to pos-array
# returns True, if estimation replaces actual recognition
def estimateTrack(frame, xp, yp, wp, hp, p, x, y, w, h, existing_pers):
# distance between last two tracking nodes
    xdiff = abs(p.pos[len(p.pos)-2][0]-xp)
    ydiff = abs(p.pos[len(p.pos)-2][1]-yp)


    if(p.oben):
        estimatedPos = (xp + int(round(0.5 * wp)) + int(round((p.movementVector[0] * xdiff)/10)),
                        yp + int(round(0.5 * hp)) + int(round((p.movementVector[1] * ydiff)/10)))
    else:
        estimatedPos = (xp + int(round(0.5 * wp)) + int(round((p.movementVector[0] * xdiff)/10)),
                        yp - int(round(0.5 * hp)) + int(round((p.movementVector[1] * ydiff)/10)))

    diffEstimatedActual = int(round(
        math.sqrt(math.pow(estimatedPos[0] - (x+int(round(0.5 * w))), 2) + math.pow(estimatedPos[1] - (y+int(round(0.5 * h))), 2))))

    print(str(diffEstimatedActual))
    if(diffEstimatedActual > 20):
        cv2.circle(frame, (int(round(x+0.5*w)), int(round(y+0.5*h))), 15, (0, 255, 0), 1)
        print("ESTIMATED")
        cv2.circle(frame, (int(round(estimatedPos[0])), int(round(estimatedPos[1]))), 15, (0, 0, 255), 1)
        if p.oben:
            existing_pers.pos.append((estimatedPos[0] - int(round(p.pos[len(p.pos) - 2][2] * 0.5)),
                                     estimatedPos[1] - int(round(p.pos[len(p.pos) - 2][3] * 0.5)),
                                     p.pos[len(p.pos) - 2][2], p.pos[len(p.pos) - 2][3]))
        else:
            existing_pers.pos.append((estimatedPos[0] - int(round(p.pos[len(p.pos) - 2][2] * 0.5)),
                                     estimatedPos[1] + int(round(p.pos[len(p.pos) - 2][3] * 0.5)),
                                     p.pos[len(p.pos) - 2][2], p.pos[len(p.pos) - 2][3]))
        return True
    else:
        return False

# draws track following the center of the frames
def drawTrack(frame, xp, yp, wp, hp, idxp, p):
    idxpos = 0
    xvector = 0         # stores after the loop the final average x-component of the movementvector
    yvector = 0         # stores after the loop the final average y-component of the movementvector
    xcenter = int(round(xp + 0.5*wp))       # x-coordinate of the center of the latest tracking point
    ycenter = int(round(yp + 0.5 * hp))     # y-coordinate of the center of the latest tracking point

    while idxpos < len(p.pos) - 1:
        xc = p.pos[idxpos][0] + int(round(p.pos[idxpos][2] * 0.5))  # x-coordinate of center of stored position
        xcn = (p.pos[idxpos + 1][0] + int(round(p.pos[idxpos + 1][2] * 0.5)))   # x-coordinate of center of point after xc
        if p.oben:
            yc = p.pos[idxpos][1] + int(round(p.pos[idxpos][3] * 0.5))  # y-coordinate of center of stored position
            ycn = (p.pos[idxpos + 1][1] + int(round(p.pos[idxpos + 1][3] * 0.5)))   # y-coordinate of center of point after yc
        else:
            yc = p.pos[idxpos][1] - int(round(p.pos[idxpos][3] * 0.5))  # y-coordinate of center of stored position
            ycn = (p.pos[idxpos + 1][1] - int(round(p.pos[idxpos + 1][3] * 0.5)))

        # adding up all pos koordinates to get one large average-vector
        xvector += xcn - xc
        yvector += ycn - yc

        cv2.circle(frame, (xc, yc), 3, (255, 255, 255), 1)
        cv2.line(frame, (xc, yc), (xcn, ycn), (255, 255, 255), 1)
        idxpos += 1
    cv2.circle(frame, (xcenter, ycenter), 3, (255, 255, 255), 1)

    # if more than one tracking point are stored, calculate a unity vector from the large one, add it to the person and draw it
    if(idxpos > 0):
        # to avoid division by zero
        if(abs(xvector)==0):
            xvector=1
        if(abs(yvector)==0):
            yvector=1
        lenvector = math.sqrt((math.pow(xvector, 2)) + (math.pow(yvector, 2)))
        xvector = int(round(((10 / lenvector)) * xvector))
        yvector = int(round(((10 / lenvector)) * yvector))
        pers = tracked_pers.pop(idxp)
        pers.movementVector = (xvector, yvector)
        tracked_pers.insert(idxp, pers)
        cv2.arrowedLine(frame, (xcenter, ycenter), (xcenter + xvector*5, ycenter + yvector*5), (0, 0, 255), 2)

# increases counter if tracked person passes trough counting_frame(red)
def countPers(yp, hp, p, idxp, oben):
    ycenter = yp + int(round(0.5 * hp))

    if(int(round(0.5*conf["resolution"][1])) < int(round(ycenter - 0.25 * hp)) and len(p.pos) > 3 and not p.counted and oben and tracked_pers):
        existing_pers = tracked_pers.pop(idxp)
        existing_pers.counted = True
        tracked_pers.insert(idxp, existing_pers)
        # global needed, because otherwise the global variable personCounter wouldn't be editable
        global personCounterRein
        personCounterRein += 1

    elif(int(round(0.5*conf["resolution"][1])) > int(round(ycenter + 0.25 * hp)) and len(p.pos) > 3 and not p.counted and not oben and tracked_pers):
        existing_pers = tracked_pers.pop(idxp)
        existing_pers.counted = True
        tracked_pers.insert(idxp, existing_pers)
        # global needed, because otherwise the global variable personCounter wouldn't be editable
        global personCounterRaus
        personCounterRaus += 1

# removes Frame if track is lost after a set period of time
def threeSecondRule(ts, idxp):
    if(ts + round(conf["del_time"]) < datetime.datetime.utcnow().second):
        if (conf["debugging"]):
            print("removing person (lost track)")
        del tracked_pers[idxp]
        return True
    else:
        return False

def pause(key):
            '''
            PRESS P TO PAUSE
            PRESS N FOR FRAME BY FRAME
            PRESS P AGAIN FOR CONTINUING NORMAL
            '''
            global stepByStep
            global nextframe
            if key == ord("q"):
                return 0
            elif key == ord("p") or stepByStep:
                stepByStep = True
                nextframe = False
                while not nextframe:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        nextframe=True
                    elif key == ord("q"):
                        return 0
                    elif key == ord("p"):
                        stepByStep = False
                        nextframe = True

def bgSubstr_1(backgroundSub, frame):

    bsMask = backgroundSub.apply(frame)
    # bsMask = cv2.erode(bsMask, None, iterations=1)
    bsMask = cv2.morphologyEx(bsMask, cv2.MORPH_CLOSE, (20, 20))
    bsMask = cv2.dilate(bsMask, (10, 10), iterations=1)
    # Gaussion Blur for larger Blobs and smoother edges
    bsMask = cv2.GaussianBlur(bsMask, (11, 11), 1)
    bsMask = cv2.medianBlur(bsMask, 5)
    return bsMask
    #cv2.imshow("bsMask", bsMask)


# actual image processing method (recognizes and counts persons)
def processIMG(capture_device):
    camera = capture_device

    backgroundSub = cv2.createBackgroundSubtractorMOG2(0, 50, True)
    avg = None

    global tsTimeLog
    stepByStep = False

    while (True):
        stime = datetime.datetime.utcnow()
        # grab the raw NumPy array representing the image and initialize
        # the timestamp and occupied/unoccupied text
        ret, frame = camera.read()
        if not ret:
            print("Rein" + str(personCounterRein))
            print("Raus" + str(personCounterRaus))
            return 0
        timestamp = datetime.datetime.now()

        # resize the frame for less computing effort
        frame = imutils.resize(frame, width=conf["resolution"][0])

        #thresh = bgSubstr_1(backgroundSub, frame)


        # converting to threshold image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 1)

        # if the average frame is None, initialize it
        if avg is None:
            print("[INFO] starting background model...")
            avg = gray.copy().astype("float")
            continue

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)
        


        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if(conf["debugging"]):
            # drawing borders for tracking
            cv2.rectangle(frame, (conf["tracking_frame"], conf["tracking_frame"]), (640 - conf["tracking_frame"], 480 - conf["tracking_frame"]), (0, 255, 0), 2)
            # drawing line for counting
            cv2.line(frame, (0, int(round(0.5*conf["resolution"][1]))), (conf["resolution"][0], int(round(0.5*conf["resolution"][1]))), (0, 0, 255), 2)

        idxp = 0

        # loop over the contours
        for idx, c in enumerate(cnts):
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)

            # if the contour is too small, ignore it
            if (cv2.contourArea(c) < conf["min_area"]):
                    continue

            # adding recognized blob (large enough to be a person) to an array

            # index of person in pers-List
            idxp = 0
            # stores information, if blob c is already a person in the array "pers"
            pers_exists = False
            for p in tracked_pers:
                if (pers_exists):
                    break
                # getting coordinates of person in array
                (xp, yp, wp, hp) = p.pos[len(p.pos)-1]

                if ( (conf["tracking_frame"] < xp < conf["resolution"][0] - conf["tracking_frame"]) and (conf["tracking_frame"] < yp < conf["resolution"][1] - conf["tracking_frame"])):
                    # if new recognized blob close to already recognized person, override position with the one of the blob
                    if (abs(xp - x) < conf["tracking_dist_max"]) and ((abs(yp - y) < conf["tracking_dist_max"] and p.oben) or (abs(yp - hp - y) < conf["tracking_dist_max"] and not p.oben)):
                        existing_pers = tracked_pers.pop(idxp)
                        existing_pers.shape = c

                        #if not estimateTrack(frame, xp, yp, wp, hp, p, x, y, w, h, existing_pers):
                        if p.oben:
                            existing_pers.pos = np.append(existing_pers.pos, [[x, y, w, h]], axis=0)
                        else:
                            existing_pers.pos = np.append(existing_pers.pos, [[x, y + h, w, h]], axis=0)

                        existing_pers.ts = datetime.datetime.utcnow().second
                        tracked_pers.insert(idxp, existing_pers)
                        pers_exists = True

                idxp += 1

            # if blob is in the recognition frame, draw frame and print index of person in array
            if ((x>conf["tracking_frame"]) and (x+w<conf["resolution"][0]-conf["tracking_frame"]) and (y>conf["tracking_frame"]) and (y+h<conf["resolution"][1]- conf["tracking_frame"])):
                # if blob doesn't fit to any persons position, add as a new person
                if (pers_exists == False):
                    if(conf["debugging"]):
                        print ("adding new person")
                    p = Person(c)
                    # when person is closer to upper
                    if((y - conf["tracking_frame"]) < (conf["resolution"][1] - conf["tracking_frame"] - (y+h))):
                        # position stored is the top left corner of the frame around the blob
                        p.pos = np.array([[x, y, w, h]])
                        p.oben = True
                    else:
                        # position stored is the bottom left corner of the frame around the blob
                        p.pos = np.array([[x, y+h, w, h]])
                        p.oben = False
                    tracked_pers.append(p)

        idxp = 0
        # loop for checking who is still in the tracking-frame, drawing and counting stuff
        for p in tracked_pers:
            (xp, yp, wp, hp) = p.pos[len(p.pos)-1]

            # position yp gets normalized to top left corner to make drawing more easily
            if not p.oben:
                yp = yp - hp
            if(conf["debugging"]):
                drawRecFrame(frame, xp, yp, wp, hp, p, idxp)
                drawTrack(frame, xp, yp, wp, hp, idxp, p)

            # if person gets removed from track, it can't be removed by threeSecondRule any more
            # if person wasn't removed, check if person should be counted
            if not (removeFromTracking(xp, yp, wp, hp, idxp)):
                if not (threeSecondRule(p.ts, idxp)):
                    countPers(yp, hp, p, idxp, p.oben)
            idxp += 1

        # draw the text and timestamp on the frame
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        if(conf["debugging"]):
            frame = imutils.resize(frame, width=640, height=480)
            thresh = imutils.resize(thresh, width=640, height=480)
            cv2.putText(frame, "Persons in frame: {}".format(str(len(tracked_pers))), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Persons Rein: {}".format(str(personCounterRein)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Persons Raus: {}".format(str(personCounterRaus)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)

        etime = datetime.datetime.utcnow()

        if (conf["debugging"] == 1):
            tdiff = 1 / conf["fps"] - ((etime - stime).microseconds / 1000000)
            if(tdiff==0):
                tdiff=1
            cv2.putText(frame, "FPS: {}".format(str(abs(int(round(1/tdiff))))), (frame.shape[1]-80, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        # check to see if the frames should be displayed to screen
        if conf["show_video"]:
            # display the security feed
            cv2.imshow("Security Feed", frame)
            cv2.moveWindow("Security Feed", 200, 200)
            cv2.imshow("Thresh", thresh)
            cv2.moveWindow("Thresh", 850, 200)
            key = cv2.waitKey(1) & 0xFF

            if(conf["debugging"]):
                pause(key)

        if(conf["lora"]):
            lora()

# using video file as input
if args.get("video", None) is not None:

    # checks if video file exists
    if os.path.isfile(str(args["video"])):
        video = cv2.VideoCapture(args["video"])
        processIMG(video)
        cv2.destroyAllWindows()
    else:
        print("Fehler beim Zugriff auf Datei! Pfad ueberpruefen!")

# using camera as input
else:
    # load chipset driver for camera module
    tmp = os.popen("ls -ltrh /dev/video*").read()
    if "/dev/video0" not in tmp:
        os.system("sudo modprobe bcm2835-v4l2")
        os.system("echo \"Camera Module has been loaded\"")
    else:
        tmp = os.system("echo \"Successfully initialized camera\"")

    camera = cv2.VideoCapture(0)
    camera.set(4,tuple(conf["resolution"])[0])
    camera.set(5,tuple(conf["resolution"])[1])

    # allow the camera to warmup, then initialize the average frame, last
    # uploaded timestamp, and frame motion counter
    print("[INFO] warming up...")
    time.sleep(conf["camera_warmup_time"])

    processIMG(camera)

