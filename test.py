import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, HypotheticalDocumentEmbedder

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


def storeDocEmbeds(file, filename):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0,)
    chunks = splitter.split_text(corpus)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings1 = OpenAIEmbeddings(openai_api_key=api_key)
    embeddings2 = HypotheticalDocumentEmbedder.from_llm(llm, embeddings1, "web_search")
    vectors = FAISS.from_texts(chunks, embeddings1, metadatas=[{"source": f"{i}-pl"} for i in range(len(chunks))])

    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)


def getDocEmbeds(file, filename):
    if not os.path.isfile(filename + ".pkl"):
        storeDocEmbeds(file, filename)

    with open(filename + ".pkl", "rb") as f:
        global vectors
        vectors = pickle.load(f)

    return vectors


def conversational_chat(vectors, query, history):
    prompt_template = """Use the following pieces of context to answer the question at the end in a deeply and extensive manner, use more than 200 words at all time to answer any questions.
    remember to always be resourceful and quote your sources with the given context in-text citation in MLA format so if the context given has a backing data supporting your response should be formatted as such:
    According to the "author name" the paper explains that ... "quote supporting your claim".
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    {context}

    Question: {question}
    Helpful Answer: """

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7), retriever=vectors.as_retriever(), chain_type="refine", return_source_documents=True, qa_prompt=QA_PROMPT)
    result = qa({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def main():
    # Replace "gpt4.pdf" with the path of the PDF file you want to process
    with open("static/data/astro_demo.pdf", "rb") as file:
        vectors = getDocEmbeds(file, "astro_demo")

    history = []

    while True:
        user_input = input("Query: ")
        if user_input.lower() == "exit":
            break

        response = conversational_chat(vectors=vectors, query=user_input, history=history)
        print("Response:", response)
        print()


if __name__ == "__main__":
    main()