import openai
import serial
import json
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up the serial connection with your Arduino
ser = serial.Serial("COM3", 9600)  # Replace "COM3" with your Arduino's COM port

# Function to send commands to GPT-3.5 Turbo
def send_gpt3_command(command):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{command}.\nTranslate the command into servo movements:",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.1,
    )
    return response.choices[0].text.strip()

# Function to send servo commands to the Arduino
def send_servo_command(servo, angle):
    command = f"{servo}:{angle}"
    ser.write(command.encode())
    print(f"Sent command: {command}")

# Main loop
while True:
    try:
        # Get the command from the user
        user_command = input("Enter a command for the robotic arm: ")

        if user_command.lower() == "exit":
            break

        # Get servo movements from GPT-3.5 Turbo
        servo_movements = send_gpt3_command(user_command)
        print(f"Servo movements: {servo_movements}")

        # Split the servo movements into individual commands
        movements = servo_movements.split(',')

        # Send servo commands to the Arduino
        for movement in movements:
            servo, angle = movement.strip().split(':')
            send_servo_command(servo, angle)

    except Exception as e:
        print(f"Error: {e}")

# Close the serial connection
ser.close()