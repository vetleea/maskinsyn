# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())

(major, minor) = cv2.__version__.split(".")[:2]
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mil": cv2.TrackerMIL_create,
    }

    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
else:
    vs = cv2.VideoCapture(args["video"])
fps = None

point_lst = []

result = cv2.VideoWriter('output.mp4',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (500, 281))
while True:

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            midpoint = (int((x+w/2)), int((y+h/2)))
            point_lst.append(midpoint)
            for midpoint in point_lst:
                cv2.circle(frame,midpoint, radius=2, color=(0, 255, 0), thickness=-1 )
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        fps.update()
        fps.stop()

        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    result.write(frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
result.release()
point_lst = np.array(point_lst)
plt.plot(point_lst[:,0], -1*point_lst[:,1])
plt.show()
