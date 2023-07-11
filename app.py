import streamlit as st 

from apikey import apikey 
import os 
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain 
from langchain.chat_models import ChatOpenAI
from template import *
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=70,
        length_function=len 
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    vectorstore = FAISS.from_texts(text=text_chunks, embeddings=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory 

    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
   
    
    set_png_as_page_bg('background.jpg')

    os.environ['OPENAI_API_KEY'] = apikey 
    print('running..')

    if "conversation"  not in st.session_state:
        st.session_state.conversation = None

    # st.set_page_config(page_title="Ask Me Anything", page_icon="books")
    st.header("Ask Me Anything")
    # user_question = st.text_input("Ask a question about your document...")
    # if user_question:
    #     handle_userinput(user_question)
    # st.write(user_template.replace("{{MSG}}","Hello robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello Human"), unsafe_allow_html=True)
    st.write(css, unsafe_allow_html=True)
    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader(
            "Upload Your PDFs here and click on 'Process'", 
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                file = open('raw.txt','w')
                file.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store 
                # vectorstore= get_vectorstore(text_chunks)

                # create conversation chain 
                # st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ =="__main__":
    main()