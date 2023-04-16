from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from pydub import AudioSegment
import openai
import os
from glob import glob

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EmbedModel = "text-embedding-ada-002"
ChatModel = "gpt-3.5-turbo"
input_folder = "static/temp_uploads"
output_folder = "static/data"

class SemanticSearcherPDF():
    def index_documents(texts, folder_path):
        pdf_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')
        ]
        loaders = [UnstructuredPDFLoader(f) for f in pdf_files]
        data = loaders.load()
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(data)
        embeddings = OpenAIEmbeddings(model=EmbedModel, openai_api_key=OPENAI_API_KEY)
        db = Chroma.from_documents(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
        return db

    def ChatSemSearch(query, db, current_temperature):
        if query:
            results = db.similarity_search(query)
            template = """You are a Bot assistant answering any questions about documents.
                You are given a question and a set of documents.
                If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
                If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.
                Use bullet points if you have to make a list, only if necessary.
                DOCUMENTS: {context}
                {chat_history}
                User:{human_input}
                Tyr:"""
            prompt = PromptTemplate(
                input_variables = ["chat_history", "query", "context"],
                template = template
            )
            memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
            chain = load_qa_chain(OpenAI(temperature=0.5, model_name=ChatModel), chain_type="stuff", memory=memory, prompt=prompt)
            chain({"input_documents": results, "human_input": query}, return_only_outputs=True)
            return (chain.memory.buffer)
    
class SemanticSearcherMP3():
    def transcribe_audio_segments(self, audio_segments, input_file):
        transcript_text = ""
        for i, segment in enumerate(audio_segments):
            print(f"Transcribing segment {i+1} of {input_file}...")
            temp_filename = f"temp_segment_{i}.mp3"
            segment.export(temp_filename, format="mp3")
            with open(temp_filename, "rb") as temp_file:
                transcript = openai.Audio.transcribe("whisper-1", temp_file)
            os.remove(temp_filename)
            transcript_text += f"Segment {i+1}:\n{transcript.text}\n\n"
        return transcript_text

    def save_transcript_to_file(self, transcript_text, output_file_path):
        with open(output_file_path, "w") as output_file:
            output_file.write(transcript_text)

    def transcribe_all_mp3_files(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        input_files = sorted(glob(os.path.join(input_folder, "*.mp3")))
        for idx, input_file in enumerate(input_files):
            audio = AudioSegment.from_file(input_file, format="mp3")
            duration_ms = 20 * 60 * 1000  # 20 minutes in milliseconds
            audio_segments = [audio[i:i+duration_ms] for i in range(0, len(audio), duration_ms)]
            transcript_text = self.transcribe_audio_segments(audio_segments, input_file)
            output_file_path = os.path.join(output_folder, f"TranscFile{idx + 1}.txt")
            self.save_transcript_to_file(transcript_text, output_file_path)
            os.remove(input_file)
            print(f"Transcription complete for {input_file}.")
        