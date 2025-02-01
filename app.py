import streamlit as st
from dotenv import load_dotenv
import os
import re
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
# HF_API_KEY = os.getenv("HF_API_KEY")
# print(HF_API_KEY)
# # Hugging Face Inference API configuration
# API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
# headers = {"Authorization": f"Bearer {HF_API_KEY}"}
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define a regex function for URL detection
def extract_urls(text):
    url_pattern = r"https?://[^\s]+"
    return re.findall(url_pattern, text) 

# Define a function to extract name of social media
def extract_social_media(prompt):
    social_media = ["LinkedIn", "Facebook", "Twitter", "Instagram", "Medium "]
    for sm in social_media:
        if sm.lower() in prompt.lower():
            return sm
    return "General"

# Define a regex function to remove URL from a text.
def extract_description(text):
    url_pattern = r"https?://[^\s]+"
    return re.sub(url_pattern, "", text)

# Summarize the reference URLs
def summarize_url_contents(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.text for p in paragraphs])
        prompt_template = """
        You are a professional content writer. Summarize the following content:

        {content}

        -  Include the main points of the content.
        - Summarize within 1000 wards.
        """
        prompt = PromptTemplate(input_variables=["content"], template=prompt_template)
        llm = ChatGroq(model_name= "llama-3.1-8b-instant")
        chain = prompt | llm
        response = chain.invoke({"content": content})
        return response.content
        # if len(content) > 1500:
        #     content = content[:1500]  # Truncate to the first 1000 characters
        # return content
    except Exception as e:
        return f"Error fetching content of {url}: {str(e)}"

# Generate blog for specified social media with description and URL reference
def generate_blog(description, references, social_media):
    try:

        llm = ChatGroq(model_name= "llama-3.1-8b-instant")

        # Define prompt template for blog generation
        prompt_template = """
        You are a professional blog writer.Using the following inputs:
        - Topic description: {description}
        - Contents from reference: {references}
        - Social media platform: {social_media}
        Create an engaging blog post with exciting title and suitable for specified social media platform. Adapt the tone and style for the platform.
        """
        prompt = PromptTemplate(
            input_variables=["description", "references", "social_media"],
            template= prompt_template
        )
        chain = prompt | llm
        response = chain.invoke({
            "description": description, 
            "references":references,
            "social_media": social_media
        })
        return response.content
    except Exception as e:
        st.error(f"Error in generating blog:{str(e)}")

# Define page config
st.set_page_config(
    page_title="PostMaker AI",
    page_icon="",
    layout="wide"
)

# Set Page headers
st.header("PostMaker AI")
st.subheader("Personal Social Media Post Maker AI Tool")

input_text = st.text_area("Enter your prompt with topic description, URLs, and social media platform:", height=200)
if st.button("Create Post"):
    if input_text.strip():
        with st.spinner("Extracting information from prompt. Please wait..."):
            # Extract URLs, description and social media
            urls= extract_urls(input_text)
            description = extract_description(input_text)
            social_media = extract_social_media(input_text)

            # Get the contents from URLs
            references = []
            for url in urls:
                content = summarize_url_contents(url)
                references.append(content)
            references = " ".join(references)

            # Generate Blogs
            blog = generate_blog(description, references, social_media)
            st.write(blog)
    else:
        st.error("Please enter your prompt.")