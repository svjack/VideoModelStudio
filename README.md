# VideoModelStudio - Genshin Impact Style Environment

Welcome to the **VideoModelStudio** project! This guide will walk you through the setup and usage of the project in a Genshin Impact-inspired style. Follow the steps below to get started.

---

## Prerequisites

Before you begin, ensure your system has the necessary dependencies installed. Run the following commands:

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
```

---

## Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/svjack/VideoModelStudio && cd VideoModelStudio
   ```

2. Install the required Python packages:

   ```bash
   conda install python=3.10
   pip install -r requirements.txt
   pip install "httpx[socks]"
   pip install moviepy==1.0.3
   ```

---

## Running the Application

To start the application, simply run:

```bash
python app.py
```

---

## Dataset Preparation

### Genshin Impact Env
```python
from moviepy.editor import VideoFileClip
from moviepy.video.fx import resize

def change_video_resolution_with_padding(input_video_path, output_video_path, target_resolution):
    """
    Resize a video to the target resolution while maintaining the aspect ratio.
    Adds black padding (letterboxing) to fill the remaining space.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        target_resolution (tuple): Target resolution as a tuple (width, height), e.g., (768, 512).
    """
    # Load the video
    video_clip = VideoFileClip(input_video_path)

    # Get the original dimensions
    original_width, original_height = video_clip.size

    # Calculate the aspect ratio of the original video
    original_aspect_ratio = original_width / original_height

    # Calculate the aspect ratio of the target resolution
    target_width, target_height = target_resolution
    target_aspect_ratio = target_width / target_height

    # Determine the new dimensions while maintaining the aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        # Video is wider than the target, so scale based on width
        new_width = target_width
        new_height = int(new_width / original_aspect_ratio)
    else:
        # Video is taller than the target, so scale based on height
        new_height = target_height
        new_width = int(new_height * original_aspect_ratio)

    # Resize the video to the new dimensions using the correct resize method
    resized_clip = video_clip.fx(resize.resize, (new_width, new_height))

    # Add black padding to fill the target resolution
    padded_clip = resized_clip.margin(
        left=(target_width - new_width) // 2,
        right=(target_width - new_width) // 2,
        top=(target_height - new_height) // 2,
        bottom=(target_height - new_height) // 2,
        color=(0, 0, 0)  # Black padding
    )

    # Save the padded video
    padded_clip.write_videofile(output_video_path, codec="libx264")

    # MoviePy automatically manages resources, so no need to manually close clips


# Example usage
input_video = "《原神》风物集短片-蒙德篇.mp4"
output_video = "《原神》风物集短片-蒙德篇-768x512.mp4"
target_resolution = (768, 512)  # Desired resolution (width x height)

change_video_resolution_with_padding(input_video, output_video, target_resolution)
```

### Pixel Video
```python
from datasets import load_dataset
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.fx import resize

def change_video_resolution_with_padding(input_video_path, output_video_path, target_resolution):
    """
    Resize a video to the target resolution while maintaining the aspect ratio.
    Adds black padding (letterboxing) to fill the remaining space.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        target_resolution (tuple): Target resolution as a tuple (width, height), e.g., (768, 512).
    """
    # Load the video
    video_clip = VideoFileClip(input_video_path)

    # Get the original dimensions
    original_width, original_height = video_clip.size

    # Calculate the aspect ratio of the original video
    original_aspect_ratio = original_width / original_height

    # Calculate the aspect ratio of the target resolution
    target_width, target_height = target_resolution
    target_aspect_ratio = target_width / target_height

    # Determine the new dimensions while maintaining the aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        # Video is wider than the target, so scale based on width
        new_width = target_width
        new_height = int(new_width / original_aspect_ratio)
    else:
        # Video is taller than the target, so scale based on height
        new_height = target_height
        new_width = int(new_height * original_aspect_ratio)

    # Resize the video to the new dimensions using the correct resize method
    resized_clip = video_clip.fx(resize.resize, (new_width, new_height))

    # Add black padding to fill the target resolution
    padded_clip = resized_clip.margin(
        left=(target_width - new_width) // 2,
        right=(target_width - new_width) // 2,
        top=(target_height - new_height) // 2,
        bottom=(target_height - new_height) // 2,
        color=(0, 0, 0)  # Black padding
    )

    # Save the padded video
    padded_clip.write_videofile(output_video_path, codec="libx264")

    # MoviePy automatically manages resources, so no need to manually close clips


def process_dataset(input_dataset, output_dir, target_res=(768, 512)):
    """
    Process each video in the dataset, resize it, and save it to the output directory.

    Args:
        input_dataset (Dataset): The dataset containing videos.
        output_dir (str): Directory to save the resized videos.
        target_res (tuple): Target resolution as a tuple (width, height), e.g., (768, 512).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each sample in the dataset
    for idx, sample in enumerate(input_dataset):
        try:
            # Generate a unique output file name
            output_path = os.path.join(output_dir, f"resized_video_{idx}.mp4")

            # Extract frames from the Decord VideoReader object
            frames = [sample["video"][i].asnumpy() for i in range(len(sample["video"]))]

            # Create a temporary video clip from the frames
            temp_clip = ImageSequenceClip(frames, fps=sample["video"].get_avg_fps())

            # Save the temporary video clip to a file
            temp_path = f"temp_{idx}.mp4"
            temp_clip.write_videofile(temp_path, verbose=False)

            # Resize the video and add padding
            change_video_resolution_with_padding(temp_path, output_path, target_res)

            # Clean up the temporary file
            os.remove(temp_path)

            print(f"Processed video {idx} and saved to {output_path}")

        except Exception as e:
            print(f"Error processing video {idx}: {str(e)}")


# Load the dataset
ds = load_dataset("svjack/test-HunyuanVideo-pixelart-videos")

# Process the training set
process_dataset(ds["train"], "resized_videos_output")

# (Optional) Process the validation and test sets
# process_dataset(ds["validation"], "resized_videos_val_output")
# process_dataset(ds["test"], "resized_videos_test_output")

!zip -r resized_videos_output.zip resized_videos_output
```

The dataset should be generated and placed in the `.data/staging` directory. Ensure your dataset is structured correctly before proceeding to training.

---

## Training the Model

To train the model, use the following command with `accelerate launch`. This configuration is optimized for mixed precision training and LoRA (Low-Rank Adaptation) fine-tuning.

```bash
accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  --num_machines=1 \
  --dynamo_backend=no \
  /home/featurize/VideoModelStudio/train.py \
  --model_name ltx_video \
  --pretrained_model_name_or_path Lightricks/LTX-Video \
  --data_root .data/training \
  --video_column videos.txt \
  --caption_column prompts.txt \
  --id_token BW_STYLE \
  --video_resolution_buckets 1x512x768 9x512x768 17x512x768 33x512x768 49x512x768 65x512x768 81x512x768 97x512x768 113x512x768 129x512x768 145x512x768 161x512x768 177x512x768 193x512x768 225x512x768 257x512x768 \
  --video_reshape_mode center \
  --caption_dropout_p 0.05 \
  --caption_dropout_technique empty \
  --flow_weighting_scheme logit_normal \
  --flow_logit_mean 0.0 \
  --flow_logit_std 1.0 \
  --flow_mode_scale 1.29 \
  --training_type lora \
  --seed 42 \
  --batch_size 1 \
  --train_epochs 70 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --checkpointing_steps 100 \
  --checkpointing_limit 10 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --lr 3e-05 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --lr_power 1.0 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.0001 \
  --epsilon 1e-08 \
  --max_grad_norm 1.0 \
  --tracker_name finetrainers \
  --output_dir .data/output \
  --report_to none \
  --nccl_timeout 1800 \
  --remove_common_llm_caption_prefixes
```

### Key Parameters Explained:
- **`--mixed_precision=bf16`**: Enables mixed precision training using bfloat16.
- **`--training_type lora`**: Uses LoRA for fine-tuning.
- **`--batch_size 1`**: Sets the batch size to 1.
- **`--train_epochs 70`**: Trains the model for 70 epochs.
- **`--lr 3e-05`**: Sets the learning rate to 3e-05.
- **`--output_dir .data/output`**: Specifies the output directory for training results.

---

## Notes

- Ensure your dataset is properly formatted and placed in the correct directory before starting the training process.
- Adjust the parameters as needed to fit your specific use case.
- For troubleshooting, refer to the project's GitHub repository or open an issue.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy your journey through the world of **VideoModelStudio**! 🌟

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

### What is this project?

VMS is a Gradio app that wraps around Finetrainers, to provide a simple UI to train AI video models on Hugging Face.

You can deploy it to a private space, and start long-running training jobs in the background.

### One-user-per-space design

Currently CMS can only support one training job at a time, anybody with access to your Gradio app will be able to upload or delete everything etc.

This means you have to run VMS in a *PRIVATE* HF Space, or locally if you require full privacy.

### Similar projects

I wasn't aware of its existence when I started my project, but there is also this open-source initiative: https://github.com/alisson-anjos/diffusion-pipe-ui

## Features

### Run Finetrainers in the background

The main feature of VMS is the ability to run a Finetrainers training session in the background.

You can start your job, close the web browser tab, and come back the next morning to see the result.

### Automatic scene splitting

VMS uses PySceneDetect to split scenes.

### Automatic clip captioning

VMS uses `LLaVA-Video-7B-Qwen2` for captioning. You can customize the system prompt if you want to.

### Download your dataset

Not interested in using VMS for training? That's perfectly fine!

You can use VMS for video splitting and captioning, and export the data for training on another platform eg. on Replicate or Fal.

## Supported models

VMS uses `Finetrainers` under the hood. In theory any model supported by Finetrainers should work in VMS.

In practice, a PR (pull request) will be necessary to adapt the UI a bit to accomodate for each model specificities.

### LTX-Video

I have tested training a LoRA model using videos, on a single A100 instance.

### HunyuanVideo

I haven't tested it yet, but in theory it should work out of the box.
Please keep in mind that this requires a lot of processing mower.

### CogVideoX

Do you want support for this one? Let me know in the comments!

## Deployment

VMS is built on top of Finetrainers and Gradio, and designed to run as a Hugging Face Space (but you can deploy it anywhere that has a NVIDIA GPU and supports Docker).

### Full installation at Hugging Face

Easy peasy: create a Space (make sure to use the `Gradio` type/template), and push the repo. No Docker needed!

That said, please see the "RUN" section for info about environement variables.

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
