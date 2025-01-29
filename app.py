import streamlit as st
from dotenv import load_dotenv
import os
import re
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
print(HF_API_KEY)
# Hugging Face Inference API configuration
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

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
        return content
    except Exception as e:
        return f"Error fetching content of {url}: {str(e)}"

# Query to Hugging Face
def huggingface_query(payload):
    try:
        response = requests.post(API_URL, headers= headers, json= payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": {str(e)}}
# Generate blog for specified social media with description and URL reference
def generate_blog(description, references, social_media):

    # Load a summarization model (Flan-T5, Falcon, etc.)
    # model_name = "google/flan-t5-base"
    # Load Falcon 7B model and tokenizer
    # model_name = "tiiuae/falcon-7b"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map= "auto",
    #     torch_dtype= "auto"
    # )

    # # Create pipeline for blog generation
    # blog_generation_pipeline = pipeline(
    #     "text-generation",
    #     model= model,
    #     tokenizer= tokenizer,
    #     max_length = 512,
    #     min_length = 50,
    #     temperature = 0.7,
    #     top_p = 0.9
    # )

    # # Integrate with langchain
    # llm = HuggingFacePipeline(pipeline= blog_generation_pipeline)

    # Define prompt template for blog generation
    prompt_template = """
    You are a professional blog writer.Using the following inputs:
    - Topic description: {description}
    - Contents from reference: {references}
    - Social media platform: {social_media}
    Create an engaging blog post suitable for specified social media platform. Adapt the tone and style for the platform.
    """
    prompt = PromptTemplate(
        input_variables=["description", "references", "social_media"],
        template= prompt_template
    )
    # Render the prompt to a plain string
    formatted_prompt = prompt.format(
        description=description,
        references=references,
        social_media=social_media
    )
    # Define LLM chain
    # chain = LLMChain(llm= llm, prompt= prompt)
    # return chain.run(description= description, references= references, social_media= social_media)
    payload = {"inputs": formatted_prompt, "parameters": {"max_length": 1024, "temperature": 0.7}}
    response = huggingface_query(payload)
    if "error" in response:
        return f"Error generating blog: {response['error']}"
    return response.get("generated_text", "Error: No text generated")

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