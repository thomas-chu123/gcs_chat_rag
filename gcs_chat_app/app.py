# app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
REQUESTS_PER_MINUTE = 10
# Initialize session_state if it's not already defined
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can Ihelp you?"}]
    st.title("📝 Q&A PDF with VertexAI")

with st.spinner('Wait for it...'):
    llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
    )
    embeddings = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)
    CONNECTION_STRING ="postgresql+psycopg2://postgres:a49>rIc@10.14.48.3:5432/vector-db"
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
                                search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=retriever, return_source_documents=True, verbose=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    result = qa({"query": prompt})
    st.session_state.messages.append({"role": "assistant", "content":
        result["result"]})
    st.chat_message("assistant").write(result["result"])