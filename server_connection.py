from flask import Flask, request, jsonify
import os
import base64
import io
from Backend import get_bootleg_score  # change to MAIN function

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '' or file.filename is None:
        return jsonify({"error": "No selected file"}), 400


#Save the file
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(os.path.join(path))

    processed_img = get_bootleg_score(path)
    processed_img = base64.b64encode(processed_img).decode('utf-8')

    # Respond with a JSON message
    return jsonify({"processed_img": processed_img})

if __name__ == 'main':
    # Update this to use the SSL certificate and key
    app.run(host='192.168.196.227', port=6969, ssl_context=('ssl/server.crt', 'ssl/server.key'))