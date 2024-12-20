import gradio as gr
import requests
from dotenv import load_dotenv
import os
from difflib import get_close_matches
import json

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

with open("responses.json", "r") as f:
    predefined_responses = json.load(f)

def query(user_input):
    if user_input in predefined_responses:
        return predefined_responses[user_input]
    
    closest_match = get_close_matches(user_input, predefined_responses.keys(), n=1, cutoff=0.6)
    if closest_match:
        return predefined_responses[closest_match[0]]
    
    prompt = f"""Context: You are an AI assistant representing Jason Appiah-Kubi He is a college student at University of Maryland Baltimore County. 
    User query: {user_input} Provide a professional and relevant answer based on Jason's background in computer science and education."""
    
    try:
        payload = {"inputs": prompt, "parameters": {"max_length": 150}}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"I apologize, but I'm having trouble generating a response. Please try rephrasing your question. Error: {str(e)}"
    
iface = gr.Interface(
    fn=query,
    inputs="text",
    outputs="text",
    title="Jason's Custom GPT Agent",
    description="Ask me about Jason's skills and experience.",
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
iface.launch(share=True)
