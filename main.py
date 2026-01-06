import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
import io
import numpy as np


import base64
from io import BytesIO
from PIL import Image

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
system_message = "You are a storyteller. Only tell stories."

# ---------------------------
# TTS function
# ---------------------------
def talker(message):
    # Call OpenAI TTS
    response = openai.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",                      # choose a different voice
    input=message,
    instructions="Speak like a storyteller with a calm voice for sleeping",
    speed=0.9                         # slightly slower for clarity
)

    return response.content

## Image generation function
def artist(context):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt = f"A gentle, dreamy illustration for a bedtime story {context}, with soft, muted colors, warm ambient lighting, cozy and calm atmosphere, featuring characters or scenery that feel sleepy and relaxed, whimsical and storybook style, highly detailed, painterly textures, cinematic composition",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

# ---------------------------
# Chat function
# ---------------------------
def chat(history):
    # Convert history for OpenAI API
    history_for_api = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role":"system", "content":system_message}] + history_for_api
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    reply = response.choices[0].message.content

    history.append({"role":"assistant", "content":reply})
    
    voice = talker(reply)

    image = artist(reply)
    return history, voice,image

# ---------------------------
# Callback to put user message in chat
# ---------------------------
def put_message_in_chatbot(message, history):
    return "", history + [{"role":"user", "content":message}]

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=400, type="messages")
        image_output = gr.Image(height=400,width=400,interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(
        put_message_in_chatbot, 
        inputs=[message, chatbot], 
        outputs=[message, chatbot]
    ).then(
        chat, 
        inputs=chatbot, 
        outputs=[chatbot, audio_output,image_output]  # None for image
    )

ui.launch()
