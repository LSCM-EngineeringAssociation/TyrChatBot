import os
import json
import openai
import io
from dotenv import load_dotenv, set_key
import requests
import webbrowser
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, render_template
import tyr_util as tyr
from enum import IntEnum
import serial
import struct
import librosa

# Flask Controls
Tyr = Flask(__name__)
Tyr.config['UPLOAD_FOLDER'] = 'static/data'
ALLOWED_EXTENSIONS = {'pdf', 'mp3'}

class Personality(IntEnum):
    NORMAL = 0
    SAFE = 1
    HAPPY = 2
    SATIRICAL = 3
    Dan = 4
    WRITING_HELPER = 5
    JOE_ROGAN = 6

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elabs_apikey = os.getenv("elabs_apikey")
elabs_voiceID = os.getenv("elabs_voice")
global current_temperature
global current_personality
global file_embeddings

# OpenAI Controls
current_temperature = 0.7
current_personality = Personality.NORMAL
file_embeddings = None

# ---------------  OPENAI FUNCTIONS  ---------------
# Set Tyr personality
with open("static/prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

messages = [
    [{"role": "system", "content": data["Prompts"]["Normal"]}],
    [{"role": "system", "content": data["Prompts"]["Safe"]}],
    [{"role": "system", "content": data["Prompts"]["Happy"]}],
    [{"role": "system", "content": data["Prompts"]["Satirical"]}],
    [{"role": "system", "content": data["Prompts"]["Dan"]}],
    [{"role": "system", "content": data["Prompts"]["WRITING_HELPER"]}],
    [{"role": "system", "content": data["Prompts"]["Joe_Rogan"]}]
]

# Auhtenticate the API key
def test_oai_key(api_key):
    url = "https://api.openai.com/v1/engines"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True
    else:
        return False
    
def test_elab_key(api_key):
    url = "https://api.elevenlabs.io/v1/user"
    headers = {
        "accept": "application/json",
        "xi-api-key": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True
    else:
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Generate response with personality 1
def generate_response(input: str, personality: Personality):
    if input:
        for i in range(len(messages)):
            messages[i].append({"role": "user", "content": input})

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages[int(personality)], temperature=current_temperature
        )
        reply = chat.choices[0].message.content
        for i in range(len(messages)):
            messages[i].append({"role": "assistant", "content": reply})

        return reply   

# ---------------  FLASK FUNCTIONS  ---------------
@Tyr.route('/')
def index():
    # Render the index page.
    return render_template('TyrPage.html')

@Tyr.route('/ask', methods=['POST'])
def ask():
    global current_personality
    conversation = str(request.get_json(force=True).get("conversation", ""))
    print("Received data:", conversation, current_personality.name)
    # Call the appropriate function based on the selected personality
    reply = generate_response(conversation, current_personality)

    print("Generated response:", reply)

    return jsonify({'text': reply})

@Tyr.route('/process_file', methods=['POST'])
def process_file():
    global file_embeddings
    file = request.files.get('file')
    # Save the file to the /static/data folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(Tyr.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Process the file using store_doc_embeds function
    file_embeddings = tyr.get_doc_embeds(filepath)
    print("done")
    return jsonify(success=True)

@Tyr.route('/get_answer', methods=['POST'])
def get_answer():
    global file_embeddings
    global current_temperature
    query = str(request.get_json(force=True).get("conversation", ""))
    history = []
    if file_embeddings:
        answer = tyr.conversational_chat(vectors=file_embeddings, query=query, history=history, temperature=current_temperature)
        return jsonify({'text': answer})
    else:
        return jsonify(error='No file has been processed yet.')

@Tyr.route('/update-openai-api-key', methods=['POST'])
def update_openai_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_oai_key(new_key):
        openai.api_key = new_key
        set_key(".env", "OPENAI_API_KEY", new_key)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})

@Tyr.route('/update-elevenlabs-api-key', methods=['POST'])
def update_elevenlabs_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_elab_key(new_key):
        os.environ["elabs_apikey"] = new_key
        set_key(".env", "elabs_apikey", new_key)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})

@Tyr.route('/update_temperature', methods=['POST'])
def update_temperature():
    data = request.get_json()
    new_temperature = float(data.get('temperature', 0.7))
    current_temperature = new_temperature
    return jsonify({'temperature': current_temperature})

@Tyr.route('/update_personality', methods=['POST'])
def update_personality():
    global current_personality
    try:
        personality_number = int(request.json['personality'])
        current_personality = Personality(personality_number)
        print(current_personality.name)
        return jsonify(personality=current_personality.name), 200
    except ValueError:
        print("Invalid personality number")
        return jsonify(error="Invalid personality number"), 400

@Tyr.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    ELABS_STAB = 0.70
    ELABS_SIMIL = 0.75
    try:
        text = request.get_json(force=True).get("text","")
        voice_id = elabs_voiceID # Replace with your desired voice ID
        api_key = elabs_apikey # Replace with your Eleven Labs API key
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': api_key,
            'Content-Type': 'application/json'
        }
        data = {
            "text": text,
            "voice_settings": {
                "stability": ELABS_STAB,
                "similarity_boost": ELABS_SIMIL
            }
        }
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            with open('temp_audio.mp3', 'wb') as f:
                f.write(response.content)

            with open('temp_audio.mp3', 'rb') as f:
                audio_data = io.BytesIO(f.read())
                y, sr = librosa.load(audio_data, sr=None)
            os.remove('temp_audio.mp3')

            # Find the highest point in the audio signal
            max_value = max(y)

            # Define the interval length in seconds
            interval_length = 0.5

            # Calculate the number of samples per interval
            samples_per_interval = int(sr * interval_length)

            # Initialize an empty array to store the relative volume values
            relative_volumes = []

            # Loop through each interval in the audio signal
            for i in range(0, len(y), samples_per_interval):
                # Calculate the maximum value in the current interval
                max_interval = max(y[i:i+samples_per_interval])
                
                # Calculate the relative volume compared to the highest point in the file
                relative_volume = max_interval / max_value
                
                # Add the relative volume to the array
                relative_volumes.append(relative_volume)

            relative_volumes_int = []

            for v in relative_volumes:
                relative_volumes_int.append(int(150 * v))

            # Open the serial port for communication
            ser = serial.Serial('COM3', 9600)

            # Send the number of data points to the Arduino Mega
            ser.write(bytes([len(relative_volumes_int)]))

            # Send the vector of integers to the Arduino Mega
            ser.write(struct.pack(f'{len(relative_volumes_int)}i', *relative_volumes_int))

            return send_file(audio_data, mimetype='audio/mpeg', as_attachment=True, download_name='response.mp3')
        else:
            print(f"Error in text_to_speech: Eleven Labs API returned {response.status_code} - {response.text}")
            return jsonify({"status": "error", "text": "Text-to-speech conversion failed"}), response.status_code
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        print(request.get_json())
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    Tyr.run()