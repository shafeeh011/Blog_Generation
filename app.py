import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import ctransformers
import ollama

def getLLama_response(input_text, no_words, blog_style):
    # Prompt template
    template = """
    Write a blog in the style of {blog_style} for the topic "{input_text}".
    Make sure it's around {no_words} words.
    """

    # Format the prompt
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    ).format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    # Send prompt to the llama3.2 model via Ollama
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "max_tokens": int(no_words) + 50  # Slight buffer
        }
    )

    return response['message']['content']


st.set_page_config(
    page_title="Gererate Blogs 🤖",
    page_icon= '🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header('Generate Blogs 🤖')

input_text = st.text_input("Enter The Blog Topic")

## create 2 more columns for additional two fields

col1,col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No.of words")
with col2:
    blog_style = st.selectbox("Blog Style for", ('Research', 'Data Science', 'Common People'), index=0)

submit = st.button("Submit")

if submit:
    st.write(getLLama_response(input_text, no_words, blog_style))