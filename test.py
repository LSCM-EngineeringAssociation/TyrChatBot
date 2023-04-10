import json

with open("prompts.json", "r") as f:
    data = json.load(f)

print(data["Prompts"]["Safe"])