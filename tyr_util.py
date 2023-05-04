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
from PyPDF2 import PdfReader
from pydub import AudioSegment
from langchain.chains import RetrievalQA
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
    with open("static/embed_data/" + filename + ".pkl", "wb") as f:
        pickle.dump(retriever, f)

def get_doc_embeds(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.isfile("static/embed_data/" + filename + ".pkl"):
        store_doc_embeds(file_path, filename)

    with open("static/embed_data/" + filename + ".pkl", "rb") as f:
        vectors = pickle.load(f)

    return vectors

def conversational_chat(vectors, query, history, temperature=0.7):
    prompt_template = """FOLLOW THESE GUIDELINES PERFECTLY:
    1)Consider the given context to identify relevant information. Use analogies, real-world applications, or storytelling to make the answer more relatable and engaging for the reader. Break down complex concepts into smaller, more manageable parts, explaining each step logically. 
    2)Address any misconceptions or misunderstandings to clarify the information effectively. Customize your approach knowing the target audience is not well versed with the topic, ensuring that your response is accessible and relevant to them. 
    3)Be sure to include appropriate humor, where applicable, to make your response more memorable and approachable. Be patient, approachable, and encourage questions, as needed, to ensure understanding.
    4)As you craft your response, always use at least 200 words to discuss the topic thoroughly. If you don't know the answer, be honest and admit that you don't know, rather than trying to fabricate a response.

    {context}

    Question: {question}
    Helpful Answer: """

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature), retriever=vectors.as_retriever(search_type="similarity", search_kwargs={"k":5}), chain_type="stuff", return_source_documents=True, qa_prompt=QA_PROMPT)
    result = qa({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


