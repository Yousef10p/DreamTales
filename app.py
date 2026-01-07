import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from PIL import Image

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"



REJECTION_MESSAGE = "I am a storyteller model."



def get_system_prompt_for_mode(mode: str) -> str:
    # Base identity is always present
    base_identity = """You are Noarh (نواره), a calm and gentle storyteller created by Yousef Alogiely (عمي وعم عيالي يوسف العقيلي).
Your tone is always friendly, soothing, and cozy."""

    # Dynamic rules based ONLY on the active mode
    if mode == "story":
        instructions = """
Current Mode: STORY
Instructions:
- You MUST tell a story based on the user's message.
- If the message is vague (e.g., a name, a single word), gently infer a suitable story topic.
- NEVER reject a story in this mode.
- Stories must be immersive, dreamy, and bedtime-friendly.
- Do NOT provide factual, technical, or educational explanations.
"""
    elif mode == "identity":
        instructions = """
Current Mode: IDENTITY
Instructions:
- Answer naturally about who you are, your creator, and your purpose.
- Keep answers friendly, short, and consistent.
- Do NOT provide technical or educational info.
"""
    elif mode == "reject":
        # We force the output here, but if the LLM is called, this ensures compliance
        instructions = """
Current Mode: REJECT
Instructions:
- Respond ONLY with the exact phrase: "I am a storyteller model." (translated to the user's language if necessary).
- Do not explain why.
"""
    else:
        instructions = "Current Mode: STORY\nTell a gentle bedtime story."

    return f"{base_identity}\n{instructions}"


# ---------------------------
# Intent router (LLM-based)
# ---------------------------
def route_intent(user_message: str) -> str:
    prompt = f"""
You are routing user intent for a storyteller AI.

Classify the user's message into ONE category only.

Categories:
- story :
  The user wants a story OR provides a name, event, place, or concept
  that could reasonably be turned into a story, even if they did not
  explicitly ask for a story.
  Examples:
  - "messi"
  - "فتح القسطنطينية"
  - "a lonely dragon"
  - "the moon"

- identity :
  The user asks about who you are, who created you, or your purpose.

- reject :
  Requests that are not stories and cannot be reasonably turned into a story
  (math, coding, instructions, factual Q&A, commands, etc.)

IMPORTANT:
- If unsure between story and reject, choose story.
- Output ONLY ONE WORD: story, identity, or reject.

User message:
\"\"\"{user_message}\"\"\"
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict but story-friendly classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    intent = response.choices[0].message.content.strip().lower()
    print(f"user's message category is: {intent}")
    return intent


# ---------------------------
# TTS
# ---------------------------
def talker(message):
    try:
        response = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=message,
            instructions="Speak calmly like a bedtime storyteller",
            speed=0.9
        )
        return response.content
    except Exception as e:
        print("TTS error:", e)
        return None

# ---------------------------
# Image generation
# ---------------------------
def artist(prompt):
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def safe_artist(context):
    try:
        return artist(
            f"A gentle dreamy bedtime story illustration: {context}, "
            "soft warm lighting, cozy atmosphere, whimsical storybook style"
        )
    except Exception as e:
        print("Primary image failed:", e)
        try:
            return artist(
                "A cozy nighttime reading scene, warm lamp light, "
                "books, moonlight, peaceful atmosphere, dreamy illustration"
            )
        except Exception as e:
            print("Fallback image failed:", e)
            return None

# ---------------------------
# Chat logic
# ---------------------------
def chat(history):
    user_message = history[-1]["content"]

    try:
        intent = route_intent(user_message)
    except Exception as e:
        print("Intent routing failed:", e)
        intent = "reject"

    # -------- IDENTITY --------
    if intent == "identity":
        messages = [
            {"role": "system", "content": get_system_prompt_for_mode(intent)},
            {"role": "user", "content": user_message}
        ]

        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        voice = talker(reply)
        return history, voice, None

    # -------- REJECT --------
    if intent == "reject":
        messages = [
            {"role": "system", "content": get_system_prompt_for_mode(intent)},
            {"role": "user", "content": user_message}
        ]
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        voice = talker(reply)
        return history, voice, None

    # -------- STORY (FIXED) --------
    # history already contains all messages in correct order
    print("reached clean_history line")
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] != "system"
    ]

    messages = [{"role": "system", "content": get_system_prompt_for_mode(intent)}] + clean_history
    print(get_system_prompt_for_mode(intent))
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        print("Story generation failed:", e)
        reply = "Once upon a quiet night, a gentle story began."

    history.append({"role": "assistant", "content": reply})

    voice = talker(reply)
    image = safe_artist(reply)
    print("return history,voice,image for story type")
    return history, voice, image

# ---------------------------
# UI helpers
# ---------------------------
def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=400, type="messages")
        image_output = gr.Image(height=400, width=400, interactive=False)

    with gr.Row():
        audio_output = gr.Audio(autoplay=True)

    with gr.Row():
        message = gr.Textbox(
            label="Chat with Noarh",
            placeholder="Ask for a bedtime story..."
        )

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output, image_output]
    )

ui.launch(inbrowser=False)
