"""
ComfyUI-InstaSD
Custom nodes for InstaSD integration with ComfyUI, including API inputs, S3 storage,
style selection, utility nodes for video and image handling, and GPT image generation.
"""

# Import mappings from both modules
from .InstaSD import NODE_CLASS_MAPPINGS as INSTA_NODE_CLASS_MAPPINGS
from .InstaSD import NODE_DISPLAY_NAME_MAPPINGS as INSTA_NODE_DISPLAY_NAME_MAPPINGS
from .gpt import NODE_CLASS_MAPPINGS as GPT_NODE_CLASS_MAPPINGS
from .gpt import NODE_DISPLAY_NAME_MAPPINGS as GPT_NODE_DISPLAY_NAME_MAPPINGS

# Merge the mappings
NODE_CLASS_MAPPINGS = {**INSTA_NODE_CLASS_MAPPINGS, **GPT_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**INSTA_NODE_DISPLAY_NAME_MAPPINGS, **GPT_NODE_DISPLAY_NAME_MAPPINGS}

# You can still import the individual classes if needed for other purposes
# but they're not required for the node registration

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

import os
import sys

# Get the directory of this file
extension_folder = os.path.dirname(os.path.realpath(__file__))

# Add the js directory to the web extensions
WEB_DIRECTORY = os.path.join(extension_folder, "js")

# This is the correct way to register web extensions in ComfyUI
if "WEB_DIRECTORY" in sys.modules[__name__].__dict__:
    pass  # Already registered
