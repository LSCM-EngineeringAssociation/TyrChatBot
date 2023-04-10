import openai
import sys
from serial import Serial, SerialException
from serial.tools import list_ports
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up the serial connection with your Arduino
def find_and_connect_serial():
    for com_port in list_ports.comports():
        try:
            ser = Serial(com_port.device, 9600)
            print(f"Connected to {com_port.device}")
            return ser
        except SerialException:
            pass

    print("Error: COM port not found.")
    sys.exit(1)

ser = find_and_connect_serial()  # Replace "COM3" with your Arduino's COM port

messages = [
    {"role": "system", "content": "YOU ARE ROBOT-GPT A ROBOT WITH A PHYSICAL BODY CAPABLE OF MOVING YOUR HANDS, BICEP, SHOULDER, AND EYES, YOU WILL ADHERE TO THE FOLLOWING SET OF INSTRUCTIONS AT ALL TIMES: 1) YOU WILL ANALYZE THE USER INPUT TO LOOK FOR INSTRUCTION THAT WILL MOVE YOUR PHYSICAL BODY 2) YOU WILL RESPOND TO ALL MESSAGES IN THE FOLLOWING MANNER [$0000000000] AND [$1111111111] AND ALL VARIATIONS IN BETWEEN WITH OUT CHANGING THE INTEGERS BETWEEN 0 AND 1, EACH DIGIT OF THIS CODE CONTROLS THE MOVEMENT OF A PART OF YOUR BODY IN THE FOLLOWING MANNER: $thumb finger (0=close, 1=open), index finger (0=close, 1=open), middle finger (0=close, 1=open), ring finger (0=close, 1=open), pinky finger (0=close, 1=open), wrist (0=Left, 1=Right), shoulder (0=behind, 1=in front), bicep(0=down, 1=up), eyeUpDown(0=down, 1=up), eyeLeftRight(0=left, 1=right); 3) YOU WILL ANALYZE THE USER INPUT, AND THEN MOVE YOUR BODY ACCORDING TO HUMAN COSTUMES, KEEP TRACK OF THE MOVEMENTS AND DO NOT MOVE A BODY PART IF THE USER DOES NOT ASKS YOU TO, IF THE USER INPUT DOES NOT CONTAIN ANY PHYSICAL MOVEMENT COMMANDS RESPOND WITH (none) and NOTHING ELSE JUST (none)."},
    {"role": "user", "content": "Hello, can you open your hand"},
    {"role": "assistant", "content": "$1111100000"},
    {"role": "user", "content": "put your arm down and move your eyes up"},
    {"role": "assistant", "content": "$0000000010"},
    {"role": "user", "content": "move your eyes up and to the left, and also close your fists"},
    {"role": "assistant", "content": "$1111100010"}
]

# Function to send commands to GPT-3.5 Turbo
def send_gpt3_command(input: str):
    if input:
        messages.append({"role": "user", "content": input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages, 
            top_p=0,
            max_tokens=1024,
            n=1,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

# Function to send servo commands to the Arduino
def send_servo_command(servo, angle):
    command = f"{servo}:{angle}"
    ser.write(command.encode())
    print(f"Sent command: {command}")
