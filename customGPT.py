import gradio as gr
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(user_input):
    payload = {"inputs": user_input}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json().get("generated_text", "Sorry, I couldn't generate a response.")

# Gradio Interface
iface = gr.Interface(
    fn=query,
    inputs="text",
    outputs="text",
    title="My Custom GPT Agent",
    description="Ask me about my skills, experience, or ideas for the role!"
)
iface.launch()
