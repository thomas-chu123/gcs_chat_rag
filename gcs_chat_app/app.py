# app.py
import os
# import tkinter as tk
# from tkinter import filedialog
from dotenv import load_dotenv
import streamlit as st
import vertexai
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI


# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_google_vertexai import ChatVertexAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAI

REQUESTS_PER_MINUTE = 10
PROJECT_ID = "ctg-rag-model-001"
LOCATION = "us-central1"
BUILD_TIME = "202501071340"
DEBUG = False
RE_WRITE = False

# def select_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_selected = filedialog.askdirectory()
#     root.destroy()
#     return folder_selected

class ChatBot:
    def __init__(self,
                 model_name: str = "gemini-1.5-flash",
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 top_k: float = 40,
                 max_tokens: int = 512,
                 retrival_k: int = 1):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.ui_interface = st
        self.retrival_k = retrival_k

        load_dotenv()
        # Set page configuration
        self.ui_interface.set_page_config(
            page_title="Q&A PDF with VertexAI",
            page_icon="📝",
            layout="wide",
        )
        # Sidebar Navigation
        self.pages = ["Chat Assistant", "Embedding Assistant"]
        self.page = self.ui_interface.sidebar.radio("Assistant Navigation", self.pages)

        # Add custom CSS to resize the sidebar
        # self.ui_interface.markdown(
        #     """
        #     <style>
        #     [data-testid="stSidebar"] {
        #         width: 400px; /* Adjust the width as needed */
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )

        if os.environ.get("stage") == 'dev':
            self.connection_string = "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
        else:
            self.connection_string = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"

        with self.ui_interface.spinner('Wait for it...'):
            # App configuration
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self.llm = ChatVertexAI(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                max_retries=3,
                # verbose=True,
                # streaming=True,
                # stop=None,
                # callbacks=[StreamingStdOutCallbackHandler()],
                # other params...
            )
            self.prompt_llm = ChatVertexAI(
                model="gemini-1.5-flash",
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                max_tokens=256,
                max_retries=3,
                verbose=True,
                streaming=True,
                stop=None,
                # callbacks=[StreamingStdOutCallbackHandler()],
                # other params...
            )
            self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=PROJECT_ID, location=LOCATION)
            self.collection_name = 'test_collection'
            self.db = PGVector.from_existing_index(
                embedding=self.embeddings,
                connection_string=self.connection_string,
                collection_name=self.collection_name,
            )
            # Initialize memory
            # self.memory = ConversationBufferMemory(memory_key="chat_history",
            #                                        output_key="result",
            #                                        return_messages=True)
            self.memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                                         output_key="result",
                                                         return_messages=True,
                                                         k=5)

            if 'memory' not in self.ui_interface.session_state:
                self.ui_interface.session_state.memory = self.memory
            else:
                self.memory = self.ui_interface.session_state.memory

            self.retriever = self.db.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": self.retrival_k}, )
            # self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
            #                                  chain_type="stuff",
            #                                  retriever=self.retriever,
            #                                  return_source_documents=True,
            #                                  memory=self.memory,
            #                                  verbose=True)
            self.qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                                  chain_type="stuff",
                                                                  retriever=self.retriever,
                                                                  return_source_documents=True,
                                                                  memory=self.memory,
                                                                  get_chat_history=lambda h: h,
                                                                  output_key='result',
                                                                  verbose=True)
            self.loader = GCSDirectoryLoader(project_name="ctg-rag-model-001",
                                             bucket="ctg-rag-model-bucket-001")
            self.ui_interface.write(f"Welcome to the Q&A PDF with VertexAI Assistant (ver: {BUILD_TIME})")
            # time.sleep(3)
            # Initialize session_state if it's not already defined
            self.ui_interface.success("Chat is ready")

    def embed_assistant(self, ):
        self.ui_interface.title("📝 Embedding Assistant")
        uploaded_files = self.ui_interface.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
        if uploaded_files is not None:
            if self.ui_interface.button("Generate Embedding"):
                for file_item in uploaded_files:
                    self.ui_interface.write(f"File Name: {file_item.name}")
                    temp_file = "./" + file_item.name
                    with open(temp_file, "wb") as file:
                        file.write(file_item.getvalue())
                        file_name = file_item.name
                    documents = PyPDFLoader(file_path=temp_file, extract_images=True).load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
                    texts = text_splitter.split_documents(documents)

                    db = PGVector.from_documents(
                        embedding=self.embeddings,
                        documents=texts,
                        connection_string=self.connection_string,
                        collection_name=self.collection_name,
                    )
                self.ui_interface.success("Embedding is done")
        else:
            self.ui_interface.info("Please upload a PDF file")

    def chat_assistant(self, ):
        result = None
        self.ui_interface.title("📝 Q&A PDF with VertexAI")
        if "messages" not in self.ui_interface.session_state:
            self.ui_interface.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, how I can help you today?", "avatar": "🤖"}
            ]

        for msg in self.ui_interface.session_state.messages:
            self.ui_interface.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])

        if prompt := self.ui_interface.chat_input():
            self.ui_interface.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👩‍🎨"})
            self.ui_interface.chat_message("user", avatar="👩‍🎨").write(prompt)
            if DEBUG:
                self.ui_interface.chat_message("assistant",avatar="🤖").write(
                    f"The memory number stored: {len(self.memory.chat_memory.messages)}")
                self.ui_interface.chat_message("assistant",avatar="🤖").write(
                    f"The session memory number stored: {len(self.ui_interface.session_state.memory.chat_memory.messages)}")

            if RE_WRITE:
                system = ("you are a experienced prompt engineer, "
                          "please state only the truth and rewrite the following text to a better prompt.")
                human = prompt
                template = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                chain = template | self.llm
                # fine-tune the user input with llm
                updated_prompt = chain.invoke({})
                updated_prompt = updated_prompt.content
                if DEBUG:
                    self.ui_interface.chat_message("assistant",avatar="🤖").write(
                        f"new converted prompt: {updated_prompt}")
                result = self.qa_chain.invoke({"question": updated_prompt})
            else:
                result = self.qa_chain.invoke({"question": prompt})

            appendix = ""
            images_dict = {}
            # Display the response
            if "source_documents" in result:
                appendix = "\n\nThe reference contents: \n"
                for id, doc in enumerate(result["source_documents"]):
                    if 'group' in doc.metadata:
                        appendix += f"\n\tHotel Group: {doc.metadata['group']}"
                    if 'region' in doc.metadata:
                        appendix += f"\n\tHotel Region: {doc.metadata['region']}"
                    if 'country' in doc.metadata:
                        appendix += f"\n\tHotel Country: {doc.metadata['country']}"
                    if 'name' in doc.metadata:
                        appendix += f"\n\tHotel Name: {doc.metadata['name']}"
                    if 'source' in doc.metadata and 'page' in doc.metadata:
                        appendix += f"\n\tDocument: {doc.metadata['source'].split('/')[-1]} in page {str(doc.metadata['page'])}\n"
                    if doc.metadata.get("type") == "image":
                        # appendix += f"\tOriginal content: {doc.metadata['original_content']}\n"
                        # self.ui_interface.image(doc.metadata["original_content"], caption=doc.page_content)
                        images_dict['content'] = doc.metadata["original_content"]
                        images_dict['caption'] = doc.page_content
            output = result["result"]
            self.ui_interface.chat_message("assistant",avatar="🤖").write(output.replace('$','\$') + "\n" + appendix)
            if images_dict:
                self.ui_interface.image(images_dict['content'], caption=images_dict['caption'])
            self.ui_interface.session_state.messages.append(
                {"role": "assistant", "content": output + "\n" + appendix, "avatar": "🤖",}
            )


def main():
    chatbot = ChatBot(model_name="gemini-pro", temperature=0.6, top_p=0.6, top_k=20, max_tokens=512, retrival_k=3)
    if chatbot.page == "Embedding Assistant":
        chatbot.embed_assistant()
    elif chatbot.page == "Chat Assistant":
        chatbot.chat_assistant()


if __name__ == "__main__":
    main()
