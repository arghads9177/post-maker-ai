# Import libraries
import time
import streamlit as st
from dotenv import load_dotenv
import os
import re
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.crawl4ai_tools import Crawl4aiTools
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

MAX_TOKENS = 6000
# Use a smaller, faster model
model = Groq(id="llama-3.1-8b-instant")

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

# Async function to fetch and summarize content
async def fetch_and_summarize(session, url, num_words= 1000):
    try:
        async with session.get(url) as response:
            content = await response.text()
            print(len(content))
            

            # # Web Crawler Agent
            # crawl_agent = Agent(
            #     name="URL Crawler Agent",
            #     model= model,
            #     tools= [Crawl4aiTools(max_length= 6000)],
            #     instructions= ["Crawl the contents of the URL"],
            #     show_tool_calls= True,
            #     markdown= True
            # )
            # print("Running Crawling Agent:")
            # response = crawl_agent.run("Read the contents of " + url)
            # content = response.content
            # Summarizer Agent
            summarizer_agent = Agent(
                name= "Web Content Summarizer",
                # model= Groq(id="llama-3.1-8b-instant"),
                model = model,
                show_tool_calls= True,
                markdown= True
            )
            prompt_template = (f"""
            You are a professional content writer. Your task is to summarize key points of the following content:
            {content}

            - Summarized content should be concise and to the point.
            - Summarized content must not be more than {num_words} words.
            - If the summarized content exceeds {num_words} words, rewrite it and shorten it to meet the word limit.
            """)
            print("Running Summarizer Agent:")
            response = summarizer_agent.run(prompt_template)
            summarized_content = response.content
            print(summarized_content)
            words = summarized_content
            if len(words) > num_words:
                summarized_content =summarized_content[:num_words]
            return summarized_content
    except Exception as e:
        return f"Error processing {url}: {e}"

# Async function to process all URLs concurrently
async def process_urls(urls, limit= 1000):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_summarize(session, url, limit) for url in urls]
        summaries = await asyncio.gather(*tasks)
        return summaries
# Summarize reference URLs
def summarize_url_contents(url, num_words= 1000):

    # Web Crawler Agent
    crawl_agent = Agent(
        name="URL Crawler Agent",
        model= model,
        tools= [Crawl4aiTools(max_length= 6000)],
        instructions= ["Crawl the contents of the URL"],
        show_tool_calls= True,
        markdown= True
    )
    print("Running Crawling Agent:")
    response = crawl_agent.run("Read the contents of " + url)
    content = response.content
    print("Crawled Content")
    # print(content)
    print(len(content))
    # response = requests.get(url)
    # soup = BeautifulSoup(response.content, "html.parser")
    # paragraphs = soup.find_all("p")
    # content = " ".join([p.text for p in paragraphs])

    # Summarizer Agent
    summarizer_agent = Agent(
        name= "Web Content Summarizer",
        # model= Groq(id="llama-3.1-8b-instant"),
        model = model,
        show_tool_calls= True,
        markdown= True
    )
    prompt_template = (f"""
    You are a professional content writer and specialist in yexy summarization. Your task is to summarize key points of the following content:
    {content}

    - Summarized content should be concise and to the point.
    - Summarized content must not be more than {num_words} words.
    - If the summarized content exceeds {num_words} words, rewrite it and shorten it to meet the word limit.
    """)
    print("Running Summarizer Agent:")
    response = summarizer_agent.run(prompt_template)
    summarized_content = response.content
    words = summarized_content
    if len(words) > num_words:
        summarized_content =summarized_content[:num_words]
    print(summarized_content)
    print(len(summarized_content))
    return summarized_content

# # Generate blog for specified social media with description and URL reference
def generate_blog(description, references, social_media):
    try:
        # Define prompt template for blog generation
        prompt_template = (f"""
        You are a professional blog writer.Using the following inputs:
        - Topic description: {description}
        - Contents from reference: {references}
        - Social media platform: {social_media}
        Create an engaging blog post with exciting title and suitable for specified social media platform. Adapt the tone and style for the platform.
        """)
        # Define Blog Writer Agent
        blog_agent = Agent(
            name="Blog Writer",
            model= model,
            # model = Groq(id= "deepseek-r1-distill-llama-70b"),
            show_tool_calls= True,
            markdown= True
        )
        response = blog_agent.run(prompt_template)
        return response.content
    except Exception as e:
        st.error(f"Error in generating blog:{str(e)}")

# Define page config
st.set_page_config(
    page_title="PostMaker AI",
    page_icon="",
    layout="wide"
)

# Streamlit App
def main():
    # Set Page headers
    st.header("PostMaker AI")
    st.subheader("Personal Social Media Post Maker AI Tool")

    input_text = st.text_area("Enter your prompt with topic description, URLs, and social media platform:", height=200)
    if st.button("Create Post"):
        if input_text.strip():
            start_time = time.time() # Start time
            with st.spinner("Extracting information from prompt. Please wait..."):
                # Extract URLs, description and social media
                urls= extract_urls(input_text)
                description = extract_description(input_text)
                social_media = extract_social_media(input_text)
            if len(urls) > 0:
                words = int(MAX_TOKENS /len(urls))
                with st.spinner("Extracting information from URLs. Please wait..."):
                #     # Get the contents from URLs
                #     for url in urls:
                #         content = summarize_url_contents(url, words)
                #         references.append(content)
                #     references = " ".join(references)
                    references = asyncio.run(process_urls(urls, words))
            with st.spinner("Writing Blog. Please wait..."):
                # Generate Blogs
                # blog = generate_blog(description, references, social_media)
                blog = generate_blog(description, references, social_media)
                st.write(blog)
                end_time = time.time() # End time
                total_time = end_time - start_time
                st.success(f"Blog generated in {total_time:.2f} seconds.")
        else:
            st.error("Please enter your prompt.")

if __name__ == "__main__":
    main()