README_WIP.md
---
title: Video Model Studio
emoji: 🎥
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: All-in-one tool for AI video training
---

# 🎥 Video Model Studio (VMS)

## Presentation

VMS is an all-in-one tool to train LoRA models for various open-source AI video models:

- Data collection from various sources
- Splitting videos into short single camera shots
- Automatic captioning
- Training HunyuanVideo or LTX-Video

## Similar projects

I wasn't aware of it when I started this project,
but there is also this: https://github.com/alisson-anjos/diffusion-pipe-ui

## Installation

VMS is built on top of Finetrainers and Gradio, and designed to run as a Hugging Face Space (but you can deploy it elsewhere if you want to).

### Full installation at Hugging Face

Easy peasy: create a Space (make sure to use the `Gradio` type/template), and push the repo. No Docker needed!

### Dev mode on Hugging Face

Enable dev mode in the space, then open VSCode in local or remote and run:

```
pip install -r requirements.txt
```

As this is not automatic, then click on "Restart" in the space dev mode UI widget.

### Full installation somewhere else

I haven't tested it, but you can try to provided Dockerfile

### Full installation in local

the full installation requires:
- Linux
- CUDA 12
- Python 3.10

This is because of flash attention, which is defined in the `requirements.txt` using an URL to download a prebuilt wheel (python bindings for a native library)

```bash
./setup.sh
```

### Degraded installation in local

If you cannot meet the requirements, you can:

- solution 1: fix requirements.txt to use another prebuilt wheel
- solution 2: manually build/install flash attention
- solution 3: don't use clip captioning

Here is how to do solution 3:
```bash
./setup_no_captions.sh
```

## Run

### Running the Gradio app

Note: please make sure you properly define the environment variables for `STORAGE_PATH` (eg. `/data/`) and `HF_HOME` (eg. `/data/huggingface/`)

```bash
python app.py
```

### Running locally

See above remarks about the environment variable.

By default `run.sh` will store stuff in `.data/` (located inside the current working directory):

```bash
./run.sh
```