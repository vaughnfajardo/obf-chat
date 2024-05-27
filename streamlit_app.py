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

def retrieve_query(query, k=10):
    return index.similarity_search(query, k=k)

def generate_response(text):
    if text[-1].isalpha():  # Ensure text ends with a period
        text += "."
    query_results = retrieve_query(text)
    response = chain.run(input_documents=query_results, question=text)
    return response

def process_input(user_input):
        if user_input[-1].isalpha():
            user_input += "."
        response = generate_response(user_input)
        return response

def main():
    st.set_page_config(page_title="OBFchat", page_icon="💙")
    st.title("OBFchat 💙")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question:", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.write_stream(process_input(user_input))

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()