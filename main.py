import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=''


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

    )   
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    
    prompt_template = """
    You are a Resume builder and suggestor.The pdf given to you is a potential candidate's resume. You are provided with the job description. 
    Please suggest the candidate changes to his resume based on job description such that he gets shortlised for the interview. 
    The candidate also are provides a question which he wants to ask regarding his Resume.\n\n
    Resume:\n {context}?\n
    Job Description:\n{job_descp}\n\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "Job Description", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(job_descp,user_question):
    embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

    ) 
    
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs,"job_descp":job_descp, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    
def main():
    st.set_page_config("Resume Builder")
    st.header("Chat with AI to Build your resume")

    
    job_descp=st.text_input("Give a Job Description for your resume")
    user_question = st.text_input("Ask a Question from the Resume Files")
    if st.button("Enter"):
        if user_question and job_descp:
            user_input(user_question, job_descp)
        else:
            st.warning("Please enter both the job description and the question.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
    