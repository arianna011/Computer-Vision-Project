from flask import Flask, request, jsonify
import os
import base64
import time
import sys
import re

# backend functionalities
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Backend.main import main
from Backend.midi_trimming import trim_midi


app = Flask(__name__)

# folders for upload and processed reponses
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PROCESSED_FOLDER = './processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route('/process-image', methods=['POST'])
def process_image():
    start_time = time.time()

    # file validation
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No valid file provided"}), 400

    # saving the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # processing the option
    option_map = {0: "MIDI", 1: "PDF"}
    option = option_map.get(int(request.form.get("option", -1)))
    if not option:
        return jsonify({"error": "Invalid output option"}), 400
    
    # elaborating the output
    response_path, play_time = main(file_path, option)

    # get author and title
    author, title = find_author_title(response_path)

    print(author, title)

    # midi trimming if needed
    output_path = os.path.join(PROCESSED_FOLDER, "trimmed.mid") if option == "MIDI" else response_path
    if option == "MIDI":
        trim_midi(response_path, output_path, play_time[0], play_time[1])
    
    # encode outpit in base64
    with open(output_path, "rb") as response_file:
        response = base64.b64encode(response_file.read()).decode('utf-8')
    
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    return jsonify({
        "output": response,
        "author": author,
        "title": title
                    }), 200


def find_author_title(file_name: str) -> tuple[str, str]:
    """Finds the name of the author and the title of the opera.

    Args:
        file_name (str): Name or path of the file. Must terminate with /p<num>.mid or /p<num>.pdf

    Returns:
        tuple[str, str]: Author and title.
    """

    txt_path = os.path.join("data", "author_title", "author_title.txt")

    regex_str = r'/p(\d+)\.(?:mid|pdf)$'
    num_opera = int(re.search(regex_str, file_name).group(1))

    with open(txt_path, 'r') as file:
        line = file.readlines()
        if num_opera >= len(line):
            return "???", "???"
        line = line[num_opera-1]
        line = line.split("\t")
        author, title = line[0], line[1]

    print(f"{num_opera=}")
    
    return author, title



if __name__ == '__main__':
    app.run(host='192.168.196.227', port=6969, ssl_context=('ssl/server.crt', 'ssl/server.key'))

