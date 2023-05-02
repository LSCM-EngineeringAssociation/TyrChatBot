from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever, PineconeHybridSearchRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainFilter
#----------------------TEST STUFF----------------------
import io
import os
import glob
import openai
import pickle
import pinecone
from PyPDF2 import PdfReader
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# AUDIO PROCESSING
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
    with open("static/embed_data/"+filename + ".pkl", "wb") as f:
        pickle.dump(retriever, f)

def get_doc_embeds(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.isfile(filename + ".pkl"):
        store_doc_embeds(file_path, filename)

    with open(filename + ".pkl", "rb") as f:
        vectors = pickle.load(f)

    return vectors

def conversational_chat(vectors, query, history, temperature=0.75):
    prompt_template = """Use the following pieces of context to answer the question at the end in a deeply and extensive manner, use more than 200 words at all time to answer any questions.
    remember to always be resourceful and quote your sources with the given context in-text citation in MLA format so if the context given has a backing data supporting your response should be formatted as such:
    According to the "author name" the paper explains that ... "quote supporting your claim"...
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    {context}

    Question: {question}
    Helpful Answer: """

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature), retriever=vectors.as_retriever(), chain_type="stuff", return_source_documents=True, qa_prompt=QA_PROMPT)
    result = qa({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


