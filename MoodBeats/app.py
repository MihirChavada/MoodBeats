from flask import Flask, render_template, request
#flask provides you with tools, libraries and technologies that allow you to build a web application
#Import necessary libraries for the program
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

#Initialize the Flask application
app = Flask(__name__)

#Set the age for cache to 1 second
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

#Initialize a dictionary to store the language and singer information
info = {}

#Load the Haar cascade XML file and the label map
haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

#Load the saved model
print("+"*50, "loadin gmmodel")
model = load_model('model.h5')

#Load the Haar cascade classifier
cascade = cv2.CascadeClassifier(haarcascade)

#Define the main page route
@app.route('/')
def index():
	
	return render_template('index.html')

#Define the route for selecting the singer
@app.route('/choose_singer', methods = ["POST"])
def choose_singer():
	info['language'] = request.form['language']
	print(info)
	return render_template('choose_singer.html', data = info['language'])

#Define the route for emotion detection
@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
	info['singer'] = request.form['singer']

	# Set a flag to indicate whether a face was found in the webcam frame
	found = False

	# Open the webcam
	cap = cv2.VideoCapture(0)

	# Keep looping until a face is detected
	while not(found):

		# Read the webcam frame
		_, frm = cap.read()

		# Convert the frame to grayscale
		gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

		# Detect faces in the grayscale frame
		faces = cascade.detectMultiScale(gray, 1.4, 1)

		# If a face is found, store the ROI and set the flag
		for x,y,w,h in faces:
			found = True
			roi = gray[y:y+h, x:x+w]
			cv2.imwrite("static/face.jpg", roi)

	# Resize the ROI to 48x48 pixels
	roi = cv2.resize(roi, (48,48))

	# Normalize the ROI
	roi = roi/255.0
	
	# Reshape the ROI to match the model input shape
	roi = np.reshape(roi, (1,48,48,1))

	# Make a prediction using the model
	prediction = model.predict(roi)

	print(prediction)

	# Get the index of the label with the highest probability
	prediction = np.argmax(prediction)
	prediction = label_map[prediction]

	cap.release()

	link  = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction}+{info['language']}+song"
	# link  = f"https://gaana.com/search/{info['singer']}+{prediction}+{info['language']}+song"
	webbrowser.open(link)

	return render_template("emotion_detect.html", data=prediction, link=link)

if __name__ == "__main__":
	app.run(debug=True)