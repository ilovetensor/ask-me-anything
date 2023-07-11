import streamlit as st 
from apikey import apikey 
import os 
from PyPDF2 import PdfReader
from apikey import apikey
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain 
from langchain.chat_models import ChatOpenAI
from template import css 
from template import bot_template, user_template
import base64
from youtube_transcript_api import YouTubeTranscriptApi

os.environ['OPENAI_API_KEY'] = apikey
st.set_page_config(page_title="Chat with multiple PDFs",
                     page_icon=":books:")
@st.cache_data
def get_image_as_base64(file):
   with open(file, 'rb') as f:
        data = f.read() 
   return base64.b64encode(data).decode() 

# PDF Handling Functions

def get_pdf_text(pdf_docs):
   text = ""
   for pdf in pdf_docs:
      pdf_reader = PdfReader(pdf)
      for page in pdf_reader.pages:
         text += page.extract_text()
   return text


def get_text_chunks(text):
   text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
   )
   chunks = text_splitter.split_text(text)
   return chunks

def get_vectorstore(text_chunks):
   embeddings = OpenAIEmbeddings()
   # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
   vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
   return vectorstore


def get_conversation_chain(vectorstore):
   llm = ChatOpenAI()
   # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

   memory = ConversationBufferMemory(
      memory_key='chat_history', return_messages=True)
   conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory
   )
   return conversation_chain

def handle_userinput(user_question):
   response = st.session_state.conversation({'question': user_question})
   st.session_state.chat_history = response['chat_history']


   for i, message in enumerate(st.session_state.chat_history):
      if i % 2 == 0:
         st.write(user_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
      else:
         st.write(bot_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)






# Youtube Link Handling Functions

def extract_id(link):
   return link[-11:]

def raw_text_from_link(link):
   video_id = extract_id(link)
   transcript = YouTubeTranscriptApi.get_transcript(video_id) 
   raw_text=""
   for i in transcript:
      raw_text += i['text'] + " "
   return raw_text

def main():
   img = get_image_as_base64('background-.jpg')
   img2 = get_image_as_base64('background.jpg')
   img3 = get_image_as_base64('bg.jpg')
   img4 = get_image_as_base64('ai.jpg')

   app_mode = st.sidebar.selectbox('Choose the App Mode',
                                 ['About App', 'Add PDF files', 
                                    'Add Youtube link'],
                                 )
   st.markdown(f"""
      <style>
      [data-testid="stSidebar"]{{
            background-color: rgba(50,50,50, 0.891);
            
      }}
      </style>

   """, unsafe_allow_html=True)
   if app_mode == "About App":
      st.title('ASK ME ANYTHING')
      st.text('');st.text('');
      with open( "title.css" ) as file:
         st.markdown( f'<style>{file.read()}</style>' , unsafe_allow_html= True)

      st.markdown(f"""
                  <div data-testid="stText" class="imhere">Watch your <span style="color: rgb(251, 39, 74);">PDFs</span> comming <span style="color: rgb(90, 255, 24)">Alive</span>... </div>
                  """, unsafe_allow_html=True)
      st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');
      
      st.markdown('''
                  
                  Thiis is just random text and this app is about the main theme that 
                  is the presentation of pdfs into the chat bot which answers the questions f
                  rom the document provided. This is a great app built by us and we are proud 
                  of it.
                  
                  Thiis is just random text and this app is about the main theme that 
                  is the presentation of pdfs into the chat bot which answers the questions f
                  rom the document provided. This is a great app built by us and we are proud 
                  of it.
                  Thiis is just random text and this app is about the main theme that 
                  is the presentation of pdfs into the chat bot which answers the questions f
                  rom the document provided. This is a great app built by us and we are proud 
                  of it.''')
      st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');st.text('');

      st.markdown(
         f"""
         
         <style>

         [data-testid="stAppViewContainer"]{{
         background-image: url("data:image/png;base64,{img}");
         background-size: cover;
         }}
         [class="main css-k1vhr4 egzxvld5"]{{
         background: rgba(0,0,0,0.5);
         }}
         [data-testid="stHeader"]{{
         background: rgba(0,0,0,0.7);
         }}
         </style>
         """ ,
      unsafe_allow_html=True)

   if app_mode == "Add PDF files":
      st.markdown(
         f"""
         
         <style>

         [data-testid="stAppViewContainer"]{{
         background-image: url("data:image/png;base64,{img2}");
         background-size: cover;
         }}
         # [class="main css-k1vhr4 egzxvld5"]{{
         # background: rgba(0,0,0,0.5);
         # }}
         [data-testid="stHeader"]{{
         background: rgba(0,0,0,0.7);
         }}
         </style>
         """ ,
      unsafe_allow_html=True)


      st.write(css, unsafe_allow_html=True)

      if "conversation" not in st.session_state:
         st.session_state.conversation = None
      if "chat_history" not in st.session_state:
         st.session_state.chat_history = None

      st.header("Chat with multiple PDFs :books:")
      user_question = st.text_input("Ask a question about your documents:")
      if user_question:
         handle_userinput(user_question)

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

               # get the text chunks
               text_chunks = get_text_chunks(raw_text)

               # create vector store
               vectorstore = get_vectorstore(text_chunks)

               # create conversation chain
               st.session_state.conversation = get_conversation_chain(
                  vectorstore)
               # print(st.session_state.conversation)




   if app_mode=='Add Youtube link':
      st.markdown(f"""<style>[data-testid="stAppViewContainer"]{{background-image: url("data:image/png;base64,{img2}");background-size: cover;}}[data-testid="stHeader"]{{background: rgba(0,0,0,0.7);}}</style>""" ,unsafe_allow_html=True)



      st.write(css, unsafe_allow_html=True)

      if "conversation" not in st.session_state:
         st.session_state.conversation = None
      if "chat_history" not in st.session_state:
         st.session_state.chat_history = None

      st.header("Chat with YouTube 📽️🔗")
      user_question = st.text_input("Ask a question about the video :")
      if user_question:
         handle_userinput(user_question)

      with st.sidebar:
         st.subheader('Youtube link')
         link = st.text_input("Paste the youtube link here")
         if st.button("Process"):
            with st.spinner("Processing"):

               # Get raw text from video
               try:
                  raw_text = raw_text_from_link(link)
                  

               except Exception as e:
                  st.text(e)


               # get the text chunks
               text_chunks = get_text_chunks(raw_text)

               # create vector store
               vectorstore = get_vectorstore(text_chunks)

               # create conversation chain
               st.session_state.conversation = get_conversation_chain(
                  vectorstore)
               # print(st.session_state.conversation)





            

if __name__=="__main__":
   main()
    

