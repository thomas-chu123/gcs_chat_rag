from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
#from langchain.vectorstores import Chroma
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import GCSDirectoryLoader
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from flask import Flask
# added as it is necessary
import os
import unstructured
import sys
llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
    project_id='chat-app-vertex-ai'
)
app = Flask(__name__)
@app.route('/', methods = ['POST', 'GET'])
def embed():
    REQUESTS_PER_MINUTE = 100
    embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)
    # load document
    loader = GCSDirectoryLoader(project_name="chat-app-vertex-ai",
            bucket="test-training-data-uqq25-i07l3j38sua09911mybb3lw2db")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    embeddings = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

    CONNECTION_STRING = "postgresql+psycopg2://postgres:a49>krIc@10.14.48.3:5432/vector-db"
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

