import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

# Sidebar UI
with st.sidebar:
    st.title('📚 Ask Your PDF!')
    st.markdown('''
    ## How It Works
    - Upload a PDF
    - Ask any question based on its content
    - Powered by LangChain & Gemini AI
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Varshini')

def main():
    st.header("🔍 PDF Q&A Chat")

    # Check Gemini API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        st.error("⚠️ Gemini API key missing. Please check your `.env` file.")
        return

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        st.success(f"✅ Uploaded: {pdf.name}")
        pdf_reader = PdfReader(pdf)

        # Extract text from PDF
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Vector store name
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.info("🧠 Loaded cached vector store.")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            st.success("📌 New vector store created and saved.")

        # User enters a custom question
        query = st.text_input("❓ Ask anything from the PDF")

        if query and st.button("Get Answer"):
            relevant_docs = vectorstore.similarity_search(query)

            try:
                # Use Gemini model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=gemini_key
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=relevant_docs, question=query)

                st.subheader("📖 Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Gemini API error: {str(e)}")

if __name__ == '__main__':
    main()
