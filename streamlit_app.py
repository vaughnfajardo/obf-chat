import json
import logging
import os
import streamlit as st

pinecone_api_key = st.secrets["PINECONE_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

from pinecone import Pinecone
pc = Pinecone(api_key=pinecone_api_key)

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

from langchain.chains.question_answering import load_qa_chain

from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore
index_name = "obfchat"
index = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo-0125", temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

def retrieve_query(query, k=10):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

def retrieve_answers(query):
    doc_search = retrieve_query(query)
    response = chain.run(input_documents=doc_search, question=query)
    return response

def generate_response(text):
    if text[-1].isalpha():
        text = text + "."
    answer = retrieve_answers(text)
    return answer

st.set_page_config(page_title="OBFchat", page_icon="💙")
st.title("OBFchat 💙")
user_input = st.text_input("Ask a question:")
if st.button('Send'):
    response = generate_response(user_input)
    st.write(response)

