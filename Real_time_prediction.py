# import the packages
from playsound import playsound
from threading import Thread
from imutils.video import VideoStream
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.models import model_from_json
import numpy as np
import imutils
import cv2

confidence = 0.5
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

def playSound():
    playsound('Ding-sound-effect.mp3')


# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream
vs = VideoStream(src=0).start()

# load the face mask model
with open("model2.json", "r") as json_file:
    loaded_model_json = json_file.read()
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        loaded_model = model_from_json(loaded_model_json)

# load the face mask weights
loaded_model.load_weights("newmMask.h5")
print("Model loaded from disk")

LIST = ["With_Mask", "Without_Mask"]


def predict(img):
    preds = loaded_model.predict(img)
    return LIST[np.argmax(preds)]


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 800 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        conf = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if conf < confidence:
            continue

        # compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:12] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        y = startY - 10 if startY - 10 > 10 else startY + 10

        # grab the face x-y and pass it throught the mask model to make a prediction
        fc = frame[startY:endY, startX:endX]
        roi = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
        # our model take a (224,224,3) input image
        roi = cv2.resize(fc, (224, 224))
        p = image.img_to_array(roi)
        p = np.expand_dims(p, axis=0)
        images = np.vstack([p])
        pred = predict(images)

        if pred == "With_Mask":
            cv2.putText(frame, pred, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 128, 0), 2)
        else:
            cv2.putText(frame, pred, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            # ding voice begin
            thread = Thread(target=playSound)
            thread.start()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
