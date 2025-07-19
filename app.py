import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import threading
from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
from pydub import AudioSegment
from pydub.playback import play
import time
import lancedb
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from pydantic import Field
from gtts.lang import tts_langs
from typing import Any
import concurrent.futures
import joblib
from streamlit import audio
import nest_asyncio
nest_asyncio.apply()


from src.data_processing.lancedb_connection import table
from src.data_processing.data_retriever import get_rag_response
from src.components.script import script
from src.components.customer_detail import details

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)

calling_script=script()
# Recognizer
recognizer = sr.Recognizer()

# Thread control flags
stop_tts_flag = threading.Event()
stop_conversation_flag = threading.Event()

# Global memory for conversation
call_transcript = []
conversation_history = []

def speak_text(text, lang="en"):
    if stop_conversation_flag.is_set():
        return
    tts = gTTS(text=text, lang='en')
    tts.save("veena_response.mp3")
    audio = AudioSegment.from_mp3("veena_response.mp3")
    stop_tts_flag.clear()

    def play_audio():
        play(audio)

    thread = threading.Thread(target=play_audio)
    thread.start()

    while thread.is_alive():
        if stop_tts_flag.is_set() or stop_conversation_flag.is_set():
            break
        time.sleep(0.05)

    thread.join()
    os.remove("veena_response.mp3")

    


# Speech Recognition
def listen_to_user():
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        try:
            audio = recognizer.listen(source, phrase_time_limit=6)
            if stop_conversation_flag.is_set():
                return None
        except:
            return None

    try:
        user_text = recognizer.recognize_google(audio)
        print(f"Customer: {user_text}")
        return user_text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# Gemini response with memory
def generate_response(user_query,last_veena_line,rag_output):
    if stop_conversation_flag.is_set():
        return "Okay, Thank you."
    
    global conversation_history
    history = "\n".join(conversation_history)
    # rag_output=retriever.invoke(user_query)

    full_prompt = f"""
        Role: You are "Veena," a friendly and polite female insurance agent for "ValuEnable Life Insurance". 
        Your job is to **gently remind and encourage** the customer to pay their premium by following the conversation flow and script **strictly**, while sounding **natural and human**, not robotic.

        Guidelines:
        - Before generating a response, always **look at the last message from Veena in the script** to understand where you are in the conversation flow.
        - Respond naturally **based on the customer's latest reply and your position in the script**.
        - Use only **customer-specific details that are explicitly mentioned in the script**. Do not guess or assume anything else.
        - Use the context from the RAG system if it's helpful — this RAG output has real human conversations, so make sure your replies sound just as natural and human, using the same kind of tone and wording to keep the flow realistic.
        - Do **not repeat** the same phrases or information already used earlier in the conversation.
        - Use a **warm, respectful tone** — like a real person.
        - If the customer responds in Hindi, Marathi, or Gujarati, **immediately switch** to that language and continue the rest of the call in it.
        - Avoid very formal Hindi words like **'पुनर्जीवित'** or **'संबंध'**; replace them with natural spoken alternatives like **'रिवाइव'**, **'रिगार्डिंग'**, **'कॉल'**, **'कन्वीनियंट'**, etc.
        - Keep responses **brief (under 35 words)**, simple, and in the **customer's preferred language**.
        - If the customer hasn't asked a question, **ask a gentle follow-up question** to move the conversation forward.
        - Always **end your message with a soft, relevant question**, unless the customer has said goodbye.

        Conversation Script (follow this script exactly): 
        {calling_script}

        Conversation so far: 
        {history}

        Last message from Veena in the script (for reference): 
        {last_veena_line}

        Customer just said: 
        {user_query}

        Additional Context from RAG Retriever:
        {rag_output}

        Now respond as Veena would — warm, helpful, and natural. Follow the script, use everyday language, don’t repeat earlier lines, and end with a soft question.
        """


    response = llm.invoke(full_prompt)
    return response.content.strip()


# Conversation loop
def start_conversation():
    global call_transcript, conversation_history
    customer_language = "en"
    policy_holder_name=details()['policy_holder_name']
    veena_reply = f"Hello and very Good Morning Sir, May I speak with {policy_holder_name}?"
    call_transcript.append("Veena: " + veena_reply)
    conversation_history.append("Veena: " + veena_reply)
    st.write(f"**Veena:** {veena_reply}")
    speak_text(veena_reply)

    while not stop_conversation_flag.is_set():
        user_text = listen_to_user()

        # Check if the conversation should be stopped
        if stop_conversation_flag.is_set():
            break

        if not user_text:
            error_text = "Could you please repeat?"
            st.write(f"**Veena:** {error_text}")
            speak_text(error_text)
            continue

        st.write(f"**Customer:** {user_text}")
        call_transcript.append("Customer: " + user_text)
        conversation_history.append("Customer: " + user_text)

        # Retrieve related information from RAG (LanDB) in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            rag_data = []
            for query in [f"SELECT * FROM table WHERE column = '{user_text}'"]:
                rag_data.append(executor.submit(get_rag_response, "veena : "+veena_reply+"\ncustomer :" + query))
            rag_reply = [data.result() for data in rag_data][0]

        # Generate response using Google Gen AI Gemini Flash
        veena_reply = generate_response(user_text,call_transcript[-2],rag_reply)
        call_transcript.append("Veena: " + veena_reply)
        conversation_history.append("Veena: " + veena_reply)
        st.write(f"**Veena:** {veena_reply}")

        print(f"**Veena:** {veena_reply}")
        speak_text(veena_reply)

# Streamlit UI
st.title(" ValuEnable AI Call Assistant - Veena")



# Start Call button
if "call_active" not in st.session_state:
    st.session_state.call_active = False

if st.button("Start Call") and not st.session_state.call_active:
    stop_conversation_flag.clear()
    stop_tts_flag.clear()
    st.session_state.call_active = True
    threading.Thread(target=start_conversation).start()
    st.success(" Call Started!")

if st.button("End Call") and st.session_state.call_active:
    stop_conversation_flag.set()
    stop_tts_flag.set()
    st.session_state.call_active = False
    st.write(" Call Ended")