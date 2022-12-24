import gradio as gr
import json
import random
import os
import requests

# Load models through the Hugging Face Inference API
# Get HuggingFace access token from secrets for increased quota.
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = ["https://thispersondoesnotexist.com/image",
           "https://api-inference.huggingface.co/models/prompthero/openjourney",
           "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"]
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load static data
with open('./static/data.json', 'r') as data_file:
    data = json.load(data_file)

# Prompts


def random_bio_prompt(name):
    introduction = random.choice(data["introductions"])
    trail = random.choice(data["trails"])
    return f"{introduction} {name} {trail}"


def random_sd_prompt():
    sample = random.sample(data["sd_prompts"], 5)
    prompts = ", ".join(sample)
    return f"mdjrny-v4 style, human, {prompts}"

# Data generating methods


def fetch_data_stylegan():
    response = requests.request("GET", API_URL[0])
    response_file_name = "stylegan.jpeg"
    with open(response_file_name, 'wb') as f:
        f.write(response.content)
    return response_file_name


def fetch_data_openjourney():
    prompts = random_sd_prompt()
    data = json.dumps(prompts)
    response = requests.request("POST", API_URL[1], headers=headers, data=data)

    if (response.status_code >= 400):
        print(
            f"Returned with status code [{response.status_code}]. Model is busy. Fetching StyleGAN2 image now...")
        print(response.content)
        return fetch_data_stylegan()

    response_file_name = "openjourney.jpeg"
    with open(response_file_name, "wb") as f:
        f.write(response.content)
    return response_file_name


def fetch_data_gptj(name):
    prompt = random_bio_prompt(name)
    data = {
        "inputs": prompt,
    }
    response = requests.request("POST", API_URL[2], headers=headers, json=data)

    if (response.status_code >= 400):
        return "❌ GPT-J 6B model is busy..."

    result = response.json()
    return result[0]["generated_text"]


def generate_image(imageModelType):
    if (imageModelType == 1):
        img = fetch_data_openjourney()
        return gr.update(value=img)
    else:
        img = fetch_data_stylegan()
        return gr.update(value=img)


def generate_name():
    return random.choice(data["names"])


def generate_city():
    return random.choice(data["cities"])


def generate_bio(name):
    # Access the GPT_J using the prompt as input
    raw_gpt_output = fetch_data_gptj(name)
    # Truncate the string until the last properly formed sentence
    trimmed_gpt_output = raw_gpt_output.rsplit('.', 1)[0]
    return f"{trimmed_gpt_output}."


def generate_profile(imageModelType):
    # Consolidate all data from the generating methods
    image = generate_image(imageModelType)
    name = generate_name()
    location = generate_city()
    bio = generate_bio(name)
    return image, name, location, bio


ProfileGen = gr.Blocks(css="#output_image{width: 420px}")
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
            This space generates an imaginary profile of a person that does not exist, with the use of open-source machine learning models.
        """)
    generate_btn = gr.Button(value="Generate", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(shape=[420, 420], elem_id="output_image")
            imageModelType = gr.Dropdown(["StyleGAN2", "Stable Diffusion"],
                                         value="StyleGAN2",
                                         type="index",
                                         label="Model")
        with gr.Column(scale=2):
            with gr.Row():
                name = gr.Textbox(label="👋 Name")
                location = gr.Textbox(label="📍 Location")
            bio = gr.Textbox(label="✨ Bio")
            generate_btn.click(fn=generate_profile, inputs=imageModelType, outputs=[
                               image, name, location, bio], api_name="generate")
    gr.Markdown("""
        <center>

          | Model            | Citation                                                                        | How it's used in this demo     |
          | ---------------- | ------------------------------------------------------------------------------- | ------------------------------ |
          | GPT-J 6B         | [Wang et al., 2021](https://github.com/kingoflolz/mesh-transformer-jax)         | 💬 Text generation for the bio |
          | StyleGAN2        | [Karras et al., 2020](https://arxiv.org/abs/1912.04958)                         | 💆‍♀️ Generate faces              |
          | Stable Diffusion | [Rombach et al., 2022](https://ommer-lab.com/research/latent-diffusion-models/) | 💆‍♀️ Generate faces              |

        </center>
        <center>
          ⚠ <span style="color:gray;font-size:11px;font-style:italic">Any resemblance to actual persons, living or dead, or actual events is purely coincidental.</span>
          <br />
          ⌛ <span style="color:gray;font-size:11px;font-style:italic">Image generation through the OpenJourney model might take a while.</span>
          </ul>
          <a href="https://gmlunesa.com">
            <img src="https://img.shields.io/badge/gmlunesa.com-E11d48.svg?&style=for-the-badge&logoColor=white" alt="gmlunesa.com"/>
          </a>
        </center>
    """)


ProfileGen.launch()
