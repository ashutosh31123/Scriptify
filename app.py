import streamlit as st 
from utils import generate_script
import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.markdown("""
    <style>
    div.stButton > button : first-child {
        background-color: #0099ff;
        color: #ffffff;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color:#FFFFFF;
    }
    </style>""", unsafe_allow_html=True)

#Creating Session state variable
if 'API_Key' not in st.session_state:
    st.session_state['API_Key']=''
    
st.title('❤️ YouTube Script Writing Tool') 

#sidebar to capture the OpenAi API Key
st.sidebar.title("😎🗝️")
st.session_state['API_Key']= st.sidebar.text_input("What's your API key?",type="password")
st.sidebar.image('./Youtube.jpg',width=300, use_column_width=True)

# Captures User Inputs
prompt = st.text_input('Please provide the topic of the video',key="prompt")  # The box for the text prompt
video_length = st.text_input('Expected Video Length 🕒 (in minutes)',key="video_length")  # The box for the text prompt
creativity = st.slider('Creativity limit ✨ - (0 LOW || 1 HIGH)', 0.0, 1.0, 0.2,step=0.1)

submit = st.button("Generate")

if submit:
    
    if st.session_state['API_Key']:
        search_result,title,script= generate_script(prompt,video_length,creativity,st.session_state['API_Key'])
        #let's generate  the script
        st.success('Hope you like this script ❤️')
        
        #Display title
        st.subheader("Title:🔥 ")
        st.write(title)
        
        #Display Video Script
        st.subheader("Your Video Script:📝")
        st.write(script)
        
        #Display Search Engine Result
        st.subheader("Check Out - DuckDuckGo Search:🔍")
        with st.expander('Show me 👀'): 
            st.info(search_result)
    else:
        st.error("Ooopssss!!! Please provide API key.....")
