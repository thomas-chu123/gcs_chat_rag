from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain_google_vertexai import VertexAI
from langchain_community.embeddings import VertexAIEmbeddings
from flask import Flask
# added as it is necessary
import os
import unstructured
import sys

llm = VertexAI(
    model_name='text-embedding-004',
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
    project_id='ctg-rag-model-001'
)
app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def embed():
    REQUESTS_PER_MINUTE = 100
    embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)
    # load document
    loader = GCSDirectoryLoader(project_name="ctg-rag-model-001",
            bucket="ctg-rag-model-bucket-001")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    embeddings = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

    CONNECTION_STRING = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"
    COLLECTION_NAME = 'test_collection'
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
    )
    return 'done embed'


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

