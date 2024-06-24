import streamlit as st
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os 
import uuid
import logging 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set page config at the very beginning
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)

# S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path = "/tmp/"
region_name = 'us-west-2'

@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name=region_name)

bedrock_client = get_bedrock_client()
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

@st.cache_data
def load_index():
    try:
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")
        return True
    except Exception as e:
        logging.error(f"Error loading index: {e}")
        return False

@st.cache_resource
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={"max_tokens_to_sample": 512})

@st.cache_resource
def get_faiss_index():
    return FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

def main():
    st.title("📚 PDF Q&A Assistant")
    st.markdown("Ask questions about the uploaded PDF document and get AI-powered answers!")

    # Sidebar for app information
    with st.sidebar:
        st.header("About")
        st.info("This app uses AI to answer questions about uploaded PDF documents. It leverages AWS Bedrock and FAISS for efficient information retrieval and natural language processing.")

    # Load index
    if not load_index():
        st.error("Failed to load the index. Please check your connection and try again.")
        return

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Question input
    question = st.text_input("Ask a question about the PDF:", key="question_input")

    # Answer button
    if st.button("Get Answer", key="answer_button"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching for an answer..."):
                try:
                    llm = get_llm()
                    faiss_index = get_faiss_index()
                    answer = get_response(llm, faiss_index, question)
                    
                    # Add to history
                    st.session_state.history.append({"question": question, "answer": answer})
                    
                    # Display the latest answer
                    st.success("Answer found!")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logging.error(f"Error getting response: {e}")

    # Display history
    if st.session_state.history:
        st.header("Question History")
        for i, qa in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {qa['question']}", expanded=(i == 0)):
                st.markdown(qa['answer'])

    # Footer
    st.markdown("---")
    st.markdown("Powered by AWS Bedrock and FAISS")

if __name__ == "__main__":
    main()