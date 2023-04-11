# TyrChatBot
Tyr - Personality - Small web app 

This is the initial implementation of a webApp to control an Inmoov Robot using GPT-3.5 and arduino.

Instructions:

1. Install Requirements.txt.
2. input your api keys in the ".envExample" file and rename it to just ".env".
3. run Tyr.py.
4. Ask away.

# Next Features

Semantic Search:
  Use Langchain and Pinecone to semantically search a long document, this will give Tyr a large memory when it comes to clusters of Data.

Robot Control:
  General idea is to create a prompt for GPT-3.5 that will output a command to control arduino or its personality.
