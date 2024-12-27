# import all the file form wedding_list folder and emdbeding the file to database.
import os
import vertexai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

def main():
    env = load_dotenv()

    PROJECT_ID = "ctg-rag-model-001"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    if os.environ.get("stage") == 'dev':
        CONNECTION_STRING = "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
    else:
        CONNECTION_STRING = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    file_folder = os.getcwd() + "/wedding_poc/"
    for root, dirs, files in os.walk(file_folder):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    documents = PyPDFLoader(file_path=os.path.join(root,file)).load()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    texts = text_splitter.split_documents(documents)

                    COLLECTION_NAME = 'test_collection'
                    db = PGVector.from_documents(
                        embedding=embeddings,
                        documents=texts,
                        connection_string=CONNECTION_STRING,
                        collection_name=COLLECTION_NAME,
                    )
                    print(f"Embedding is done with {os.path.join(root,file)}")
                except Exception as e:
                    print(f"Error in embedding {os.path.join(root,file)}")
                    print(e)


if __name__ == "__main__":
    main()