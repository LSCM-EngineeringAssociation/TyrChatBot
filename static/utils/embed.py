from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import pinecone

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

def load_pdfs_from_folder(folder_path):
    pdf_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')
    ]
    loaders = [UnstructuredPDFLoader(f) for f in pdf_files]
    return loaders

def process_pdf(loader):
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

def initialize_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

def index_documents(texts, index_name):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings)
    return docsearch

def answer_query(docsearch, query):
    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query, include_metadata=True)
    result = chain.run(input_documents=docs, question=query)
    return result

def SemSearch(query):
    pdf_folder_path = "static/data"
    loaders = load_pdfs_from_folder(pdf_folder_path)
    initialize_pinecone()
    for loader in loaders:
        texts = process_pdf(loader)
        docsearch = index_documents(texts)
        result = answer_query(docsearch, query)
    return result
