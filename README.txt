Install packages in requirements.txt, 
cd to the directory where requirements.txt is located.
run: pip install -r requirements.txt in your shell.

and then install sklearn-contrib-lightning
run: conda install -c conda-forge sklearn-contrib-lightning 

based on https://github.com/JamesLuoau/Self-Driving-Car-Vehicle-Detection

deep neural network approach using YOLO
- much faster than the "classical" support vector machine approach using HOG + OpenCV
- vastly better generalisation (so performs better in real world data) due to nature of DNN aquiring better abstraction about the features as network grows deeper
- each layer learns something different
- lower accuracy but sufficient
- see details on source page, but the key advantage is that, YOLO, as name suggests, only look once and evaluates a whole frame at once, 
whereas SVM approach requires a method called Sliding Window Search, which splits a frame into many, many small squares with high overlap (75%), 
slide each square across the image and make predictions on each location

Works really well, except only picks up motorbikes from a distance. At close range it would detect it as a person.
Assume front facing camera (as in demo videos)
analysing a 31s video (or 949 frames) took my RSA machine under 13 minutes

Requires an input of video in mp4 format
Run main.py to analyse your video.
yolo folder contains all the support files for yolo model.
object_detect_yolo.py is the object detection algorithm using yolo model.