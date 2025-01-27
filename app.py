import streamlit as st
from dotenv import load_dotenv
import os
import re
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define a regex function for URL detection
def extract_urls(text):
    url_pattern = r"https?://[^\s]+"
    return re.findall(url_pattern, text) 

# Define a regex function to remove URL from a text.
def remove_urls(text):
    url_pattern = r"https?://[^\s]+"
    return re.sub(url_pattern, "", text)

# Load a summarization model (Flan-T5, Falcon, etc.)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map= "auto",
    torch_dtype= "auto"
)

# Create pipeline for summarization
summarization_pipeline = pipeline(
    "summarization",
    model= model,
    tokenizer= tokenizer,
    max_length = 150,
    min_length = 50,
    length_penalty = 2.0,
    num_beams = 4
)

# Integrate with langchain
llm = HuggingFacePipeline(pipeline= summarization_pipeline)

# Define prompt template for summarization anfd key point extraction
prompt = PromptTemplate(
    input_variables=["text"],
    template= """
    Summarize the following text and extract the key points:
    {text}

    Exclude any URLs from the summarization
    """
)
# Define page config
st.set_page_config(
    page_title="PostMaker AI",
    page_icon="",
    layout="wide"
)

# Set Page headers
st.header("PostMaker AI")
st.subheader("Personal Social Media Post Maker AI Tool")

input_text = st.text_area("Enter your content along with reference URLs")
if st.button("Create Post"):
    if input_text.strip():
        with st.spinner("Extracting URL. Please wait..."):
            urls= extract_urls(input_text)
            st.write(urls)