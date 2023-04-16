import os
import json
import openai
import io
from dotenv import load_dotenv
import requests
import webbrowser
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
import speech_recognition as sr
from enum import IntEnum

Tyr = Flask(__name__)

class Personality(IntEnum):
    NORMAL = 0
    SAFE = 1
    HAPPY = 2
    SATIRICAL = 3
    HIGHLY_SATIRICAL = 4
    WRITING_HELPER = 5
    JOE_ROGAN = 6

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elabs_apikey = os.getenv("elabs_apikey")
elabs_voiceID = os.getenv("elabs_voice")
global current_temperature
global current_personality

#OpenAI Controls
current_temperature = 0.7
current_personality = Personality.NORMAL

# Upload controls
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "txt", "pdf"}
Tyr.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set Tyr personality
with open("static/prompts.json", "r") as f:
    data = json.load(f)

messages = [
    [{"role": "system", "content": data["Prompts"]["Normal"]}],
    [{"role": "system", "content": data["Prompts"]["Safe"]}],
    [{"role": "system", "content": data["Prompts"]["Happy"]}],
    [{"role": "system", "content": data["Prompts"]["Satirical"]}],
    [{"role": "system", "content": data["Prompts"]["Highly_Satirical"]}],
    [{"role": "system", "content": data["Prompts"]["WRITING_HELPER"]}],
    [{"role": "system", "content": data["Prompts"]["Joe_Rogan"]}]
]

# Auhtenticate the API key
def test_api_key(api_key):
    openai.api_key = api_key
    test = openai.Completion.create(
        engine="text-ada-001", prompt="test", max_tokens=5, n=1)
    if 'Invalid API key' in str(test):
        return False
    else:
        return True

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

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

@Tyr.route('/update-api-key', methods=['POST'])
def update_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_api_key(new_key):
        openai.api_key = new_key
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
            os.remove('temp_audio.mp3')

            return send_file(audio_data, mimetype='audio/mpeg', as_attachment=True, download_name='response.mp3')
        else:
            print(f"Error in text_to_speech: Eleven Labs API returned {response.status_code} - {response.text}")
            return jsonify({"status": "error", "text": "Text-to-speech conversion failed"}), response.status_code
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        print(request.get_json())
        return jsonify({"error": str(e)}), 400
    
@Tyr.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(Tyr.config["UPLOAD_FOLDER"], filename))
            # Add your own processing logic here
            return "File uploaded and saved.", 200
        else:
            return "File type not allowed", 400
    
if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    Tyr.run()