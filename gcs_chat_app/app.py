# app.py
import os
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
import streamlit as st
import vertexai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_google_vertexai import ChatVertexAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAI


REQUESTS_PER_MINUTE = 10
env = load_dotenv()

PROJECT_ID = "ctg-rag-model-001"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# App configuration
st.set_page_config(
    page_title="Q&A PDF with VertexAI",
    page_icon="📝",
    layout="wide",
)

# Add custom CSS to resize the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 400px; /* Adjust the width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_response(prompt: str):
    pass

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


# Side bar Navigation
pages = ["Chat Assistant", "Embedding Assistant"]
page = st.sidebar.radio("Assistant Navigation", pages)

if page == "Embedding Assistant":
    st.title("📝 Embedding Assistant")
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files is not None:
        if st.button("Gernerate Embedding"):
            for file_item in uploaded_files:
                st.write(f"File Name: {file_item.name}")
                temp_file = "./" + file_item.name
                with open(temp_file, "wb") as file:
                    file.write(file_item.getvalue())
                    file_name = file_item.name
                if os.environ.get("stage") == 'dev':
                    CONNECTION_STRING = "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
                    # For local development
                    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
                    documents = PyPDFLoader(file_path=temp_file).load()
                else:
                    CONNECTION_STRING = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"
                    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
                    loader = GCSDirectoryLoader(project_name="ctg-rag-model-001",
                                                bucket="ctg-rag-model-bucket-001")

                    documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)

                COLLECTION_NAME = 'test_collection'
                db = PGVector.from_documents(
                    embedding=embeddings,
                    documents=texts,
                    connection_string=CONNECTION_STRING,
                    collection_name=COLLECTION_NAME,
                )
            st.success("Embedding is done")
    else:
        st.info("Please upload a PDF file")

elif page == "Chat Assistant":
    # Initialize session_state if it's not already defined
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
        st.title("📝 Q&A PDF with VertexAI")

    with st.spinner('Wait for it...'):
        if os.environ.get("stage") == 'dev':
            # For local development
            # llm = OpenAI(model="gpt-3.5-turbo-instruct",
            #          temperature=0.3,
            #          max_tokens=256,
            #          top_p=0.8,
            #          verbose=True,
            #          streaming=True
            # )
            # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            CONNECTION_STRING = "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
        else:
            CONNECTION_STRING = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"

        llm = ChatVertexAI(
            model="gemini-1.5-flash",
            temperature=0.6,
            top_p=0.8,
            top_k=40,
            max_tokens=512,
            max_retries=3,
            verbose=True,
            # streaming=True,
            stop=None,
            # callbacks=[StreamingStdOutCallbackHandler()],
            # other params...
        )
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=PROJECT_ID, location=LOCATION)

        COLLECTION_NAME = 'test_collection'
        db = PGVector.from_existing_index(
            embedding=embeddings,
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
        )
        st.success('Chat is ready')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How I can help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        retriever = db.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=retriever,
                                         return_source_documents=True,
                                         verbose=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        result = qa.invoke({"query": prompt})
        appendix = "\n\nThe reference content can be found in the below sources documents: \n"
        # Display source documents
        if "source_documents" in result:
            # st.session_state.messages.append({"role": "assistant", "content": "Source Documents:"})
            # st.chat_message("assistant").write("Source Documents:")
            for id, doc in enumerate(result["source_documents"]):
                # st.session_state.messages.append({"role": "assistant", "content": doc.page_content})
                appendix += f"\n\tReference document name: {doc.metadata['source']} in page {str(doc.metadata['page'])}\n"
        output = result["result"] + appendix
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.chat_message("assistant").write(output)
