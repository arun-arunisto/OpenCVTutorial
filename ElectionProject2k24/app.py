# app.py
import os
import base64
from flask import Flask, render_template, request

app = Flask(__name__)

# Create the 'images' folder if it doesn't exist
images_folder = 'images'
os.makedirs(images_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    # Get the base64-encoded image data from the request
    image_data = request.form['image_data'].split(',')[1]

    # Decode the base64 data and save the image
    image_path = os.path.join(images_folder, f'captured_image.png')
    with open(image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))

    return 'Image saved successfully'

if __name__ == '__main__':
    app.run(debug=True)
