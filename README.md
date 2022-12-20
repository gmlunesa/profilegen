---
title: ProfileGen
emoji: 🤖
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 3.14.0
app_file: app.py
pinned: false
license: mit
---

# 🔮 ProfileGen

<a href="https://huggingface.co/spaces/lunesa/ProfileGen">
  <img src="https://img.shields.io/badge/🤗%20Visit%20Demo-8B5CF6.svg?&style=for-the-badge&logoColor=white" alt="Visit Space"/>
</a>
<a href="https://gmlunesa.com/notox/#/about">
  <img src="https://img.shields.io/badge/gmlunesa.com-E11d48.svg?&style=for-the-badge&logoColor=white" alt="gmlunesa.com"/>
</a>

## About

ProfileGen generates an imaginary profile of a person that does not exist, with the use of open-source machine learning models. This project is inspired by the first iteration of the [Methical app](https://github.com/gmlunesa/methical-frontend). The interface is bootstrapped with Gradio.

## Machine Learning models

| Model            | Citation                                                                        | How it's used in this demo     |
| ---------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| GPT-J 6B         | [Wang et al., 2021](https://github.com/kingoflolz/mesh-transformer-jax)         | 💬 Text generation for the bio |
| StyleGAN2        | [Karras et al., 2020](https://arxiv.org/abs/1912.04958)                         | 💆‍♀️ Generate faces              |
| Stable Diffusion | [Rombach et al., 2022](https://ommer-lab.com/research/latent-diffusion-models/) | 💆‍♀️ Generate faces              |

## Stack

| <img src="https://raw.githubusercontent.com/yurijserrano/Github-Profile-Readme-Logos/master/programming%20languages/python.svg" width="100" height="100" alt="React"> |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## Notes

- ⌛ Image generation through the Stable Diffusion (OpenJourney) model might take a while.
- 💆‍♀️ If the Stable Diffusion (OpenJourney) model is busy, the image will be sourced from the StyleGAN API.
- ⚠ Any resemblance to actual persons, living or dead, or actual events is purely coincidental.

## License

[MIT 🌱 Fully open-source](https://github.com/gmlunesa/ProfileGen/blob/main/LICENSE)

---

Made with 💫✨ by [Goldy Mariz Lunesa @gmlunesa](https://gmlunesa.com)
