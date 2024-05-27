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

llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0.4)
chain=load_qa_chain(llm,chain_type="stuff")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="OBFchat", page_icon="💙")
st.title("OBFchat 💙")

# Display chat messages from history
for message in st.session_state.messages:
    with st.container():
        st.write(f"{message['role']}: {message['content']}")

def generate_response(text):
    if text[-1].isalpha():  # Ensure text ends with a period
        text += "."
    query_results = retrieve_query(text)
    response = chain.run(input_documents=query_results, question=text)
    return response

def retrieve_query(query, k=10):
    return index.similarity_search(query, k=k)

# Process user input and update UI
def process_input(user_input):
    if user_input:
        # Ensure user_input ends with a period for consistency in processing
        if user_input[-1].isalpha():
            user_input += "."
        response = generate_response(user_input)
        # Append user and assistant responses to chat history

        with st.chat_message("OBFchat"):
            st.write(response)


# User input interface
user_input = st.chat_input("Ask a question:", key="user_input")

if st.button('Send'):
    process_input(user_input)
    st.session_state.user_input = ""  # Clear input after processing
    with st.chat_message("You"):
        st.write(user_input)