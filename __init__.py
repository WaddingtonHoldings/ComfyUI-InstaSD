from .InstaSD import InstaCBoolean, InstaCText, InstaCTextML, InstaCInteger, InstaCFloat, InstaCSeed, InstaCSaveImageToS3, InstaCLoadImageFromS3, InstaCLoraLoader, InstaPromptStyleSelector, InstaPromptMultipleStyleSelector

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
