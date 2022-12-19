import gradio as gr
import json
import random
import os

# Load models through the Hugging Face Inference API
# Get HuggingFace access token from secrets for increased quota.
HF_TOKEN = os.environ.get("HF_TOKEN")
gpt_j = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")

# Load static data
with open('./static/data.json', 'r') as data_file:
    data = json.load(data_file)

# Prompts


def random_bio_prompt(name):
    introduction = random.choice(data["introductions"])
    trail = random.choice(data["trails"])
    return f"{introduction} {name} {trail}"

# Data generating methods


def generate_name():
    return random.choice(data["names"])


def generate_city():
    return random.choice(data["cities"])


def generate_bio(name):
    # Access the GPT_J using the prompt as input
    raw_gpt_output = gpt_j(random_bio_prompt(name))
    # Truncate the string until the last properly formed sentence
    trimmed_gpt_output = raw_gpt_output.rsplit('.', 1)[0]
    return f"{trimmed_gpt_output}."


def generate_profile():
    # Consolidate all data from the generating methods
    name = generate_name()
    location = generate_city()
    bio = generate_bio(name)
    return name, location, bio, gr.update(visible=True)


ProfileGen = gr.Blocks()
with ProfileGen:
    gr.Markdown(
        """
          <h1 align="center">
            ProfileGen
          </h1>
          <center>
            🤖🤖🤖
          </center>
          <center>
            This space generates an imaginary profile of a person that does not exist, with the use of machine learning models.
          </center>
          <center style="color:gray;font-size:11px;font-style:italic">
            ⚠ Any resemblance to actual persons, living or dead, or actual events is purely coincidental.
          </center>
          <center>
            <a href="https://gmlunesa.com">
              <img src="https://img.shields.io/badge/gmlunesa.com-E11d48.svg?&style=for-the-badge&logoColor=white" alt="gmlunesa.com"/>
            </a>
          </center>
        """)
    generate_btn = gr.Button(value="Generate", variant="primary")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="👋 Name")
        with gr.Column():
            location = gr.Textbox(label="📍 Location")
    bio = gr.Textbox(label="✨ Bio")
    generate_btn.click(fn=generate_profile, inputs=None, outputs=[
                       name, location, bio], api_name="generate_text")


ProfileGen.launch()
