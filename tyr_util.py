import os
import openai
import glob
from pydub import AudioSegment
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, HypotheticalDocumentEmbedder, ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# AUDIO PROCESSING

async def transcribe_audio_segments(self, audio_segments, input_file):
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

async def save_transcript_to_file(self, transcript_text, output_file_path):
        with open(output_file_path, "w") as output_file:
            output_file.write(transcript_text)

async def transcribe_all_mp3_files(self, input_folder, output_folder):
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

# PDF PROCESSING
#embeddings_filter = EmbeddingsFilter(embeddings=base_embeddings, similarity_threshold=0.5)
#vectors = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=docsearch)
def store_doc_embeds(file):
    loader = PyPDFLoader(file)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = loader.load_and_split(text_splitter=splitter)
    base_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search") #Doesnt work with embeddings filtering for some reason I don't understand
    docsearch = Chroma.from_documents(documents=pages, embedding=embeddings , metadatas=[{"source": f"{i}-pl"} for i in range(len(pages))])
    return docsearch

def conversational_chat(retriever, query, history):
    chat_openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = RetrievalQAWithSourcesChain.from_chain_type(chat_openai, chain_type="refine", retriever=retriever.as_retriever())
    result = chain({"question": query, "chat_history": history}, return_only_outputs=True)
    history.append((query, result["answer"]))
    return result["answer"]