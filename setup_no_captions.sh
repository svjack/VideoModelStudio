#!/usr/bin/env bash

python -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements_without_flash_attention.txt

# if you require flash attention, please install it manually for your operating system

# you can try this:
# python -m pip install wheel setuptools flash-attn --no-build-isolation --no-cache-dir