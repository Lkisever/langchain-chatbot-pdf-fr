import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from htmlTemplates import bot_template, user_template, css

from transformers import pipeline

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    
    embeddings = HuggingFaceInstructEmbeddings(model_name = "dangvantuan/sentence-camembert-base")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):
    
    llm = OpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:9999/v1",
    model_name="/mnt/d/AI/models/vigogne-2-7b-chat/"
    )

     # Define the system message template
    system_template = """Tu es un expert juridique qui assiste des magistrats.
    Si tu ne connais pas la réponse tu réponds que tu ne sais pas.
    
    CONTEXT:
    {context}
  
    QUESTION: 
    {question}
  
    ANSWER:
    """

    # Create the chat prompt templates
    messages = [
    SystemMessagePromptTemplate.from_template(system_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title='Parlez avec vos PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Parlez avec vos PDFs :books:')
    question = st.text_input("Demandez n'importe quoi à vos documents : ")

    if question:
        handle_user_input(question)
    

    with st.sidebar:
        st.subheader("Upload de documents: ")
        pdf_files = st.file_uploader("Choisissez vos fichiers PDFs et cliquez sur OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Traitement des PDFs en cours..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                

                # Create Vector Store
                
                vector_store = get_vector_store(text_chunks)
                st.write("Traitement des PDFs terminé")

                # Create conversation chain

                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
