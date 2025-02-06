import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Ensure the API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY environment variable not set. Please set it and restart the app.")
    st.stop()

st.title("PDF Chatbot")

st.write("Upload a PDF document, and then ask questions about its content.")

# File uploader for PDFs
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    st.info("Processing PDF...")
    
    try:
        # Load and split the PDF into pages (or use further splitting if needed)
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load_and_split()

        # Create embeddings for the document chunks
        embeddings = OpenAIEmbeddings()

        # Build a vector store (FAISS) for similarity search
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever()

        # Initialize the language model with a deterministic output (temperature=0)
        llm = OpenAI(temperature=0)

        # Set up the RetrievalQA chain using the "stuff" chain type
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        st.success("PDF processed! Now, you can ask questions about its content.")

        # Text input for the user's query
        query = st.text_input("Ask a question about the PDF:")

        if st.button("Submit Query") and query:
            with st.spinner("Generating answer..."):
                # Run the QA chain to get an answer
                answer = qa_chain.run(query)
            st.write("**Answer:**", answer)
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
