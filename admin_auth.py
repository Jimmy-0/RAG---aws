import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import streamlit as st
import os 
import uuid
import logging
import hashlib
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

region_name = 'us-west-2'
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=region_name) 
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username: str, password: str) -> bool:
    if username in USERS and USERS[username] == hash_password(password):
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

def login_form():
    with st.form("login_form"):
        st.write("Please log in to access the admin panel:")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if authenticate(username, password):
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.experimental_rerun()

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, splitted_docs):
    vectorstore_faiss = FAISS.from_documents(splitted_docs, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    # Upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

def process_pdf(uploaded_file):
    request_id = get_unique_id()
    st.write(f"Request Id: {request_id}")
    saved_file_name = f"{request_id}.pdf"
    with open(saved_file_name, mode="wb") as w:
        w.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader(saved_file_name)
    pages = loader.load_and_split()

    st.write(f"Total Pages: {len(pages)}")

    # Split Text
    splitted_docs = split_text(pages, 1000, 200)
    st.write(f"Splitted Docs Length: {len(splitted_docs)}")

    st.write("Creating the Vector Store")
    try:
        res = create_vector_store(request_id, splitted_docs)
        if res:
            st.write("Successfully created the Vector Store")
    except Exception as e:
        st.error(f"Error creating the Vector Store: {e}")

def protected_view():
    st.write(f"Welcome, {st.session_state.username}!")
    st.write("This is the protected admin view.")
    
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("Processing PDF...")
        process_pdf(uploaded_file)

    if st.button("Logout"):
        logout()

def main():
    init_session_state()

    if not st.session_state.authenticated:
        login_form()
    else:
        protected_view()

if __name__ == "__main__":
    main()