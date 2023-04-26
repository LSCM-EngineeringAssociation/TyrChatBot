import os
import openai
import glob
import io
import pinecone
from pydub import AudioSegment
import pickle
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever, PineconeHybridSearchRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.chains import RetrievalQAWithSourcesChain, HypotheticalDocumentEmbedder, ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV')  # next to api key in console
)

# PDF PROCESSING
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
#embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search") #Doesnt work with embeddings filtering for some reason I don't understand
#embeddings_filter = EmbeddingsFilter(embeddings=base_embeddings, similarity_threshold=0.5)
#vectors = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=docsearch)
def store_doc_embeds(file, filename):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
    chunks = splitter.split_text(corpus)

    base_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    retriever = FAISS.from_texts(texts=chunks, embedding=base_embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(chunks))]) # metadatas=[{"source": f"{i}-pl"} for i in range(len(pages))]
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(retriever, f)

def get_doc_embeds(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.isfile(filename + ".pkl"):
        store_doc_embeds(file_path, filename)

    with open(filename + ".pkl", "rb") as f:
        vectors = pickle.load(f)

    return vectors

def conversational_chat(retriever, query, history):
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6), chain_type="refine", retriever=retriever.as_retriever(), return_source_documents=True)
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def main():
    # Replace "gpt4.pdf" with the path of the PDF file you want to process
    inpath = "static/data/astro_demo.pdf"
    vectors = get_doc_embeds(inpath)

    history = []

    while True:
        user_input = input("Query: ")
        if user_input.lower() == "exit":
            break

        response = conversational_chat(vectors, user_input, history)
        print("Response:", response)
        print()


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

if __name__ == "__main__":
    main()