from flask import Flask, request, jsonify
import werkzeug 
import json
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
app = Flask(__name__)
CORS(app)
@app.route('/upload', methods = ["POST"])
def upload():
    if(request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/" + filename)
        API_URL = "https://autoderm.firstderm.com/v1/query"

    # set sensitive data as environment variables
    API_KEY = os.getenv("oEpwcN1fJAROayGDESL5hVAOCEmGGMuvjrzU-rMSw9k")

    # open the test image and read the bytes
    with open(r"uploadedimages\acne.jpg", "rb") as f:
        image_contents = f.read()

    # send the query
    response = requests.post(
        API_URL,
        headers={"Api-Key": API_KEY},
        files={"file": image_contents},
        params={"language": "en", "model": "autoderm_v3_0"},
    )

    # get the JSON data returned
    data = response.json()
    print("HERE: ++!+!+!")
    print(data)

    # Check if the 'predictions' key exists in the response
    if 'predictions' in data:
        # Get the predictions
        predictions = data["predictions"]
        print(predictions)
    else:
        print("No 'predictions' key in the response.")

    return data
        

import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def find_predominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    predominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    return predominant_color

def is_color_change(pixel, predominant_color, threshold=50):
    pixel = np.float32(pixel)
    predominant_color = np.float32(predominant_color)
    color_difference = np.linalg.norm(pixel - predominant_color)
    return color_difference > threshold

@app.route('/color', methods=['POST'])
def detect_color_change():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    predominant_color = find_predominant_color(hsv_image)
    threshold = 50
    height, width, _ = image.shape
    color_change_count = 0

    for y in range(height):
        for x in range(width):
            pixel = hsv_image[y, x]
            if is_color_change(pixel, predominant_color, threshold):
                color_change_count += 1

    total_pixels = height * width
    color_change_percentage = (color_change_count / total_pixels) * 100

    if color_change_percentage > 5:
        result = {'message': 'Significant color change detected'}
    else:
        result = {'message': 'Color change is not significant'}

    return jsonify(result)

def detect_pores(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_pore_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_pore_area]
    detected_pores = []
    for contour in filtered_contours:
        approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
        detected_pores.append(approx)
    return detected_pores

@app.route('/pores', methods=['POST'])
def detect_pores_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_path = "temp_image.jpg"
    image_file.save(image_path)

    detected_pores = detect_pores(image_path)
    num_pores = len(detected_pores)

    return jsonify({'DetectedPores': detected_pores})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
