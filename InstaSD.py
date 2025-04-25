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
from .utils import node_helpers
import hashlib

input_dir = folder_paths.get_input_directory()

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

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "InstaSD-Utility"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_dir,video)
        return (video_path,)

class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "InstaSD-Utility"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

class InstaLoadImageLocal:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files),)},
                }

    CATEGORY = "InstaSD"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class InstaLoadImageWithMask:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                "hidden": {
                    "mask_image": "MASK_IMAGE",
                }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    
    def load_image(self, image, mask_image=None):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        # Process main image
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            # If no custom mask is provided, use alpha channel from main image
            if mask_image is None:
                if 'A' in i.getbands():
                    # Get alpha channel and convert to proper format
                    # Invert the alpha values: 1.0 - alpha
                    alpha = 1.0 - np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    
                    # Create a 3-channel mask image (H,W,3) with same dimensions as original
                    mask_array = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.float32)
                    # Fill all channels with inverted alpha values
                    mask_array[:,:,0] = alpha
                    mask_array[:,:,1] = alpha
                    mask_array[:,:,2] = alpha
                    
                    # Convert to tensor with batch dimension [1,H,W,3]
                    mask_image_tensor = torch.from_numpy(mask_array)[None,]
                else:
                    # Create an empty mask with same dimensions as the image
                    mask_array = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.float32)
                    mask_image_tensor = torch.from_numpy(mask_array)[None,]
            
            output_images.append(image)
            if mask_image is None:
                output_masks.append(mask_image_tensor)

        # Process custom mask image if provided
        if mask_image is not None:
            # If mask_image is a tensor (from mask editing), use it directly
            if isinstance(mask_image, torch.Tensor):
                # Ensure mask has correct dimensions
                if len(mask_image.shape) == 2:  # Single channel mask
                    h, w = mask_image.shape
                    mask_rgb = torch.zeros((3, h, w), dtype=torch.float32, device=mask_image.device)
                    mask_rgb[0] = mask_image
                    mask_rgb[1] = mask_image
                    mask_rgb[2] = mask_image
                    mask_tensor = mask_rgb.unsqueeze(0)  # Add batch dimension
                elif len(mask_image.shape) == 3 and mask_image.shape[0] == 1:  # Batch of single channel
                    _, h, w = mask_image.shape
                    mask_rgb = torch.zeros((1, 3, h, w), dtype=torch.float32, device=mask_image.device)
                    mask_rgb[0, 0] = mask_image[0]
                    mask_rgb[0, 1] = mask_image[0]
                    mask_rgb[0, 2] = mask_image[0]
                    mask_tensor = mask_rgb
                else:
                    # Assume it's already in the right format
                    mask_tensor = mask_image
                
                output_masks.append(mask_tensor)
            else:
                # If mask_image is a string (file path), load it
                try:
                    mask_path = folder_paths.get_annotated_filepath(mask_image)
                    mask_img = node_helpers.pillow(Image.open, mask_path)
                    
                    for i in ImageSequence.Iterator(mask_img):
                        i = node_helpers.pillow(ImageOps.exif_transpose, i)
                        
                        if i.mode == 'I':
                            i = i.point(lambda i: i * (1 / 255))
                        
                        # Convert to grayscale if it's not already
                        if i.mode != 'L':
                            mask = i.convert("L")
                        else:
                            mask = i
                        
                        # Resize mask to match the main image dimensions if needed
                        if mask.size[0] != w or mask.size[1] != h:
                            mask = mask.resize((w, h), Image.LANCZOS)
                        
                        # Convert to numpy array and normalize
                        mask_array = np.array(mask).astype(np.float32) / 255.0
                        
                        # Create a 3-channel mask image
                        mask_rgb = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.float32)
                        mask_rgb[:,:,0] = mask_array
                        mask_rgb[:,:,1] = mask_array
                        mask_rgb[:,:,2] = mask_array
                        
                        # Convert to tensor
                        mask_tensor = torch.from_numpy(mask_rgb)[None,]
                        output_masks.append(mask_tensor)
                        
                        # Only use the first frame of the mask image
                        break
                except Exception as e:
                    print(f"Error loading mask image: {e}")
                    # Create an empty mask with same dimensions as the image
                    mask_array = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.float32)
                    mask_tensor = torch.from_numpy(mask_array)[None,]
                    output_masks.append(mask_tensor)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            if len(output_masks) > 1:
                output_mask = torch.cat(output_masks, dim=0)
            else:
                # If we have a single mask but multiple images, repeat the mask
                output_mask = output_masks[0].repeat(len(output_images), 1, 1, 1)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0] if output_masks else torch.zeros((1, 3, h, w), dtype=torch.float32)

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image, mask_image=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
                
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, mask_image=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

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
    "InstaPromptMultipleStyleSelector": InstaPromptMultipleStyleSelector,
    "LoadVideo": LoadVideo,
    "PreViewVideo": PreViewVideo,
    "InstaLoadImageLocal": InstaLoadImageLocal,
    "InstaLoadImageWithMask": InstaLoadImageWithMask
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
    "InstaPromptMultipleStyleSelector": "InstaSD - Multiple Style Selctor",
    "LoadVideo": "InstaSD - LoadVideo Utility Node",
    "PreViewVideo": "InstaSD - PreviewVideo Utility Node",
    "InstaLoadImageLocal": "InstaSD - Load image from local folder",
    "InstaLoadImageWithMask": "InstaSD API Input - Load Image With Mask"
}
