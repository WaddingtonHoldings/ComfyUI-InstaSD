import sys
import io
import torch
import numpy as np
import boto3
from PIL import Image, ImageSequence, ImageOps
from datetime import datetime
import folder_paths
import comfy.utils
import os
import json

STYLES_PATH = os.path.join('/ComfyUI/styles', 'styles.json')
WEBUI_STYLES_FILE = os.path.join('/ComfyUI/styles', 'styles.csv')

if  os.path.exists(WEBUI_STYLES_FILE):

    print(f"Importing styles from `{WEBUI_STYLES_FILE}`.")

    import csv

    styles = {}
    with open(WEBUI_STYLES_FILE, "r", encoding="utf-8-sig", newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt = row.get("prompt") or row.get("text", "") # Old files
            negative_prompt = row.get("negative_prompt", "")
            styles[row["name"]] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt
            }

    if styles:
        if not os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, "w", encoding='utf-8') as f:
                json.dump(styles, f, indent=4)

    del styles

    print(f"Styles import complete.")

else:
    print(f"Styles file `{WEBUI_STYLES_FILE}` does not exist. Place it under /ComfyUI/styles and restart Comfy.")

class InstaCBoolean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)

    FUNCTION = "execute"

    def execute(self, boolean=True):
        return (boolean,)


class InstaCText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "execute"

    def execute(self, string=""):
        return (string,)


class InstaCTextML:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "execute"

    def execute(self, string=""):
        return (string,)


class InstaCInteger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 1,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    FUNCTION = "execute"

    def execute(self, int=True):
        return (int,)


class InstaCFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 1,
                            "min": -sys.float_info.max,
                            "max": sys.float_info.max,
                            "step": 0.01}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)

    FUNCTION = "execute"

    def execute(self, float=True):
        return (float,)
    
class InstaCSeed:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 1,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1}),
            }
        }

    CATEGORY = "InstaSD" + "/API_inputs"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    FUNCTION = "execute"

    def execute(self, int=True):
        return (int,)
    

def awss3_save_file(client, bucket, key, buff):
    client.put_object(
            Body = buff,
            Key = key, 
            Bucket = bucket)

def awss3_load_file(client, bucket, key):
    outfile = io.BytesIO()
    client.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    return outfile

def awss3_init_client(region="us-east-1", ak=None, sk=None, session=None):
    client = None
    if (ak == None and sk == None) and session == None:
        client = boto3.client('s3', region_name=region)
    elif (ak != None and sk != None) and session == None:
        client = boto3.client('s3', region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk)
    elif (ak != None and sk != None) and session != None:
        client = boto3.client('s3', region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk, aws_session_token=session)
    else:
        client = boto3.client('s3')
    return client

class InstaCSaveImageToS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "images": ("IMAGE",), 
                             "region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "session_token": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"})
                             },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save_image_to_s3"
    CATEGORY = "InstaSD" + "/S3"
    OUTPUT_NODE = True

    def save_image_to_s3(self, images, region, aws_ak, aws_sk, session_token, s3_bucket, pathname, prompt=None, extra_pnginfo=None):
        client = awss3_init_client(region, aws_ak, aws_sk, session_token)
        results = list()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            filename = f"{pathname}_{timestamp}_{batch_number}.png"
            awss3_save_file(client, s3_bucket, filename, img_byte_arr.getvalue())
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })
        return { "ui": { "images": results } }

class InstaCLoadImageFromS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "session_token": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"})
                             } 
                }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_s3"
    CATEGORY = "InstaSD" + "/S3"

    def load_image_from_s3(self, region, aws_ak, aws_sk, session_token, s3_bucket, pathname):
        client = awss3_init_client(region, aws_ak, aws_sk, session_token)
        img = Image.open(awss3_load_file(client, s3_bucket, pathname))
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    
class InstaCLoraLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        # Always load from disk
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class InstaPromptStyleSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        style_list = []
        if os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, "r") as f:
                if len(f.readlines()) != 0:
                    f.seek(0)
                    data = f.read()
                    styles = json.loads(data)
                    for style in styles.keys():
                        style_list.append(style)
        if not style_list:
            style_list.append("None")
        return {
            "required": {
                "style": (style_list,),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("positive_string", "negative_string")
    FUNCTION = "load_style"

    CATEGORY = "WAS Suite/Text"

    def load_style(self, style):

        styles = {}
        if os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, 'r') as data:
                styles = json.load(data)
        else:
            print(f"The styles file does not exist at `{STYLES_PATH}`. Unable to load styles! Have you imported your AUTOMATIC1111 WebUI styles?")

        if styles and style != None or style != 'None':
            prompt = styles[style]['prompt']
            negative_prompt = styles[style]['negative_prompt']
        else:
            prompt = ''
            negative_prompt = ''

        return (prompt, negative_prompt)

class InstaPromptMultipleStyleSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        style_list = []
        if os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, "r") as f:
                if len(f.readlines()) != 0:
                    f.seek(0)
                    data = f.read()
                    styles = json.loads(data)
                    for style in styles.keys():
                        style_list.append(style)
        if not style_list:
            style_list.append("None")
        return {
            "required": {
                "style1": (style_list,),
                "style2": (style_list,),
                "style3": (style_list,),
                "style4": (style_list,),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("positive_string", "negative_string")
    FUNCTION = "load_style"

    CATEGORY = "WAS Suite/Text"

    def load_style(self, style1, style2, style3, style4):
        styles = {}
        if os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, 'r') as data:
                styles = json.load(data)
        else:
            print(f"The styles file does not exist at `{STYLES_PATH}`. Unable to load styles! Have you imported your AUTOMATIC1111 WebUI styles?")
            return ('', '')

        # Check if the selected styles exist in the loaded styles dictionary
        selected_styles = [style1, style2, style3, style4]
        for style in selected_styles:
            if style not in styles:
                print(f"Style '{style}' was not found in the styles file.")
                return ('', '')

        prompt = ""
        negative_prompt = ""

        # Concatenate the prompts and negative prompts of the selected styles
        for style in selected_styles:
            prompt += styles[style]['prompt'] + " "
            negative_prompt += styles[style]['negative_prompt'] + " "

        return (prompt.strip(), negative_prompt.strip())

NODE_CLASS_MAPPINGS = {
    "InstaCBoolean": InstaCBoolean,
    "InstaCText": InstaCText,
    "InstaCInteger": InstaCInteger,
    "InstaCFloat": InstaCFloat,
    "InstaCTextML": InstaCTextML,
    "InstaCSeed": InstaCSeed,
    "InstaCSaveImageToS3": InstaCSaveImageToS3,
    "InstaCLoadImageFromS3": InstaCLoadImageFromS3,
    "InstaCLoraLoader": InstaCLoraLoader,
    "InstaPromptStyleSelector": InstaPromptStyleSelector,
    "InstaPromptMultipleStyleSelector": InstaPromptMultipleStyleSelector
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InstaCBoolean": "InstaSD API Input - Boolean",
    "InstaCText": "InstaSD API Input - String",
    "InstaCInteger": "InstaSD API Input - Integer",
    "InstaCFloat": "InstaSD API Input - Float",
    "InstaCTextML": "InstaSD API Input - Multi Line Text",
    "InstaCSeed": "InstaSD API Input - Seed",
    "InstaCSaveImageToS3": "InstaSD S3 - Save Image",
    "InstaCLoadImageFromS3": "InstaSD S3 - Load Image",
    "InstaCLoraLoader": "InstaSD API Input - Lora Loader",
    "InstaPromptStyleSelector": "InstaSD - Style Selctor",
    "InstaPromptMultipleStyleSelector": "InstaSD - Multiple Style Selctor"
}
