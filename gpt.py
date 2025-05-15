import base64
import io
import os
import math
from inspect import cleandoc

import numpy as np
import requests
import torch
from PIL import Image
import os
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.utils import common_upscale
from .apis import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
)
from .apis.client import ApiEndpoint, HttpMethod, SynchronousOperation

import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def read_user_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def downscale_input(image):
    samples = image.movedim(-1, 1)
    # downscaling input images to roughly the same size as the outputs
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s


def validate_and_cast_response(response):
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise Exception("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors = []

    # Process each image in the data array
    for image_data in data:
        image_url = image_data.url
        b64_data = image_data.b64_json

        if not image_url and not b64_data:
            raise Exception("No image was generated in the response")

        if b64_data:
            img_data = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_data))

        elif image_url:
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                raise Exception("Failed to download the image")
            img = Image.open(io.BytesIO(img_response.content))

        img = img.convert("RGBA")

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        # Add to list of tensors
        image_tensors.append(img_tensor)

    return torch.stack(image_tensors, dim=0)


class GPTImage1Generate(ComfyNodeABC):
    """
    Generate images using OpenAI's GPT-Image-1 model.
    
    This node allows you to:
    - Generate images from text prompts
    - Edit existing images with text instructions
    - Inpaint parts of images using masks
    - Control quality, size, and other generation parameters
    - Use your own API endpoint and authentication
    
    The node supports both image generation from scratch and image editing modes.
    When an input image is provided, it switches to editing mode automatically.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:

        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image 1",
                    },
                ),
            },
            "optional": {
                "model": (
                    IO.STRING,
                    {
                        "default": "gpt-image-1",
                        "display": "string",
                        "tooltip": "GPT Image Model",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["low", "medium", "high"],
                        "default": "low",
                        "tooltip": "Image quality, affects cost and generation time.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Return image with or without background",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                        "default": "auto",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
                "moderation": (
                    IO.COMBO,
                    {
                        "options": ["low", "auto"],
                        "default": "low",
                        "tooltip": "Moderation level",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "External API Nodes"
    
    # Enhanced description that will show up in the UI
    DESCRIPTION = """Generate or edit images using OpenAI's GPT-Image-1 model.

This node connects to OpenAI's API (or a compatible endpoint) to generate images from text prompts or edit existing images.

Key features:
• Text-to-image generation
• Image editing with text instructions
• Inpainting with masks
• Customizable quality and size settings
• Support for transparent backgrounds
• Multiple image generation in one call

To use this node:
1. Enter your prompt describing the image you want
2. Optionally connect an image to edit and/or a mask for inpainting
3. Configure quality, size, and other parameters
4. Set your API credentials in your InstaSD account under settings (You need to restart the instance after)
"""
    
    # You can also add tooltips to the node itself
    NODE_TOOLTIP = "Generate or edit images using OpenAI's GPT-Image-1 model"

    def api_call(
        self,
        prompt,
        seed=0,
        quality="low",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        moderation="low",
        api_base=None,
        auth_token=None,
        model=None,
    ):

        config = read_user_config()
        API_BASE = config.get("api_base", "")
        AUTH_TOKEN = config.get("auth_token", "")

        # If model is None, use the default model
        if model is None:
            model = "gpt-image-1"
        path = "images/generations"
        request_class = OpenAIImageGenerationRequest
        img_binaries = []
        mask_binary = None
        files = []

        if api_base is None or api_base == "":
            # Print "No api_base found, trying to get it from settings."
            print(f"No api_base found, trying to get it from settings.")
            api_base = API_BASE

        if auth_token is None or auth_token == "":
            print(f"No auth_token found, trying to get it from settings.")
            auth_token = AUTH_TOKEN

        if image is not None:
            path = "images/edits"
            request_class = OpenAIImageEditRequest

            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_input(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                img_binary = img_byte_arr
                img_binary.name = f"image_{i}.png"

                img_binaries.append(img_binary)
                if batch_size == 1:
                    files.append(("image", img_binary))
                else:
                    files.append(("image[]", img_binary))

        if mask is not None:
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            mask_binary = mask_img_byte_arr
            mask_binary.name = "mask.png"
            files.append(("mask", mask_binary))

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                size=size,
                moderation=moderation,
            ),
            files=files if files else None,
            api_base=api_base,
            auth_token=auth_token,
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response)
        return (img_tensor,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GPTImage1Generate": GPTImage1Generate,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage1Generate": "InstaSD - OpenAI GPT Image 1",
}