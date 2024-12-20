import gradio as gr
import requests
from dotenv import load_dotenv
import os
from difflib import get_close_matches
import json

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

with open("responses.json", "r") as f:
    predefined_responses = json.load(f)

def query(user_input):
    closest_match = get_close_matches(user_input, predefined_responses.keys(), n=1, cutoff=0.6)
    if closest_match:
        return predefined_responses[closest_match[0]]
  
    payload = {"inputs": f"User query: {user_input}. Provide a professional and concise answer."}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("generated_text", "Sorry, I couldn't generate a response.")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    
iface = gr.Interface(
    fn=query,
    inputs="text",
    outputs="text",
    title="My Custom GPT Agent",
    description="Ask me about my skills, experience, or ideas for the role!",
    examples=[
        ["Tell me about yourself."],
        ["What are your skills?"],
        ["How would you approach this role?"],
        ["What experience do you have?"],
        ["What is your understanding of learning engineering?"],
        ["What are adult learning strategies?"],
        ["How can AI-driven solutions help in education?"],
    ],
)
iface.launch()
