from scipy import optimize,stats,interpolate
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import math
import json
from dotenv import load_dotenv
import os

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import google.generativeai as genai

def configure():
   load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "model", "parts": "Great to meet you. I am assuming the role of a certified financial advisor and whatever recommendations is based on your preferred portfolio. You can trust me to answer your queries such as 1. How is my portfolio looking? 2. What improvements can I make to my current portfolio? 3. Risks in my portfolio? 4. Views on the companies fundamentals / macroeconomic trends?"},
    ]

# Initialize genai model
API_KEY = os.getenv('API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history= st.session_state.messages + [{"role": "user", "parts": 'Do not ask me for my portfolio allocation again. From your knowledge and updated information, complement the allocation and data provided below'\
                                           +str(st.session_state.diagnosis)}]
)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "parts": prompt})


# Display assistant response in chat message container
import time
def stream_data(stream):
    for s in stream:
      yield s.text
      time.sleep(0.02)

# Page 2 configuration
st.title("Ask ChatAI for advice")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"])

while prompt != None:
  with st.chat_message("model"):
      stream = chat.send_message(prompt, stream=True)
      response = st.write_stream(stream_data(stream))
  st.session_state.messages.append({"role": "model", "parts": response})
  prompt = None

with st.sidebar:
  st.download_button('Download Chat History', json.dumps(st.session_state.messages), 'chat.json')

## With careful consideration of my portfolio and the fundamentals behind each company, evaluate my portfolio and suggest ways to improve.
## Now explain how the recent US elections and how Trump, as well as the macroeconomy today, will bring about risks and opportunities to my portfolio
## What are the sentiments based on the recent news articles?
## What are some macroeconomic trends and figures to consider in the upcoming months, and what does it indicate for the economy and the portfolio?
## Is there any part of the fundamental data that is worrying? List down all the risks involved in the portfolio