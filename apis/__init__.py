# generated by datamodel-codegen:
#   filename:  https://api.comfy.org/openapi
#   timestamp: 2025-04-23T15:56:33+00:00

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AnyUrl, BaseModel, Field, confloat, conint

class Customer(BaseModel):
    createdAt: Optional[datetime] = Field(
        None, description='The date and time the user was created'
    )
    email: Optional[str] = Field(None, description='The email address for this user')
    id: str = Field(..., description='The firebase UID of the user')
    name: Optional[str] = Field(None, description='The name for this user')
    updatedAt: Optional[datetime] = Field(
        None, description='The date and time the user was last updated'
    )


class Error(BaseModel):
    details: Optional[List[str]] = Field(
        None,
        description='Optional detailed information about the error or hints for resolving it.',
    )
    message: Optional[str] = Field(
        None, description='A clear and concise description of the error.'
    )


class ErrorResponse(BaseModel):
    error: str
    message: str

class ImageRequest(BaseModel):
    aspect_ratio: Optional[str] = Field(
        None,
        description="Optional. The aspect ratio (e.g., 'ASPECT_16_9', 'ASPECT_1_1'). Cannot be used with resolution. Defaults to 'ASPECT_1_1' if unspecified.",
    )
    color_palette: Optional[Dict[str, Any]] = Field(
        None, description='Optional. Color palette object. Only for V_2, V_2_TURBO.'
    )
    magic_prompt_option: Optional[str] = Field(
        None, description="Optional. MagicPrompt usage ('AUTO', 'ON', 'OFF')."
    )
    model: str = Field(..., description="The model used (e.g., 'V_2', 'V_2A_TURBO')")
    negative_prompt: Optional[str] = Field(
        None,
        description='Optional. Description of what to exclude. Only for V_1, V_1_TURBO, V_2, V_2_TURBO.',
    )
    num_images: Optional[conint(ge=1, le=8)] = Field(
        1, description='Optional. Number of images to generate (1-8). Defaults to 1.'
    )
    prompt: str = Field(
        ..., description='Required. The prompt to use to generate the image.'
    )
    resolution: Optional[str] = Field(
        None,
        description="Optional. Resolution (e.g., 'RESOLUTION_1024_1024'). Only for model V_2. Cannot be used with aspect_ratio.",
    )
    seed: Optional[conint(ge=0, le=2147483647)] = Field(
        None, description='Optional. A number between 0 and 2147483647.'
    )
    style_type: Optional[str] = Field(
        None,
        description="Optional. Style type ('AUTO', 'GENERAL', 'REALISTIC', 'DESIGN', 'RENDER_3D', 'ANIME'). Only for models V_2 and above.",
    )


class Datum(BaseModel):
    is_image_safe: Optional[bool] = Field(
        None, description='Indicates whether the image is considered safe.'
    )
    prompt: Optional[str] = Field(
        None, description='The prompt used to generate this image.'
    )
    resolution: Optional[str] = Field(
        None, description="The resolution of the generated image (e.g., '1024x1024')."
    )
    seed: Optional[int] = Field(
        None, description='The seed value used for this generation.'
    )
    style_type: Optional[str] = Field(
        None,
        description="The style type used for generation (e.g., 'REALISTIC', 'ANIME').",
    )
    url: Optional[str] = Field(None, description='URL to the generated image.')


class Code(Enum):
    int_1100 = 1100
    int_1101 = 1101
    int_1102 = 1102
    int_1103 = 1103


class Code1(Enum):
    int_1000 = 1000
    int_1001 = 1001
    int_1002 = 1002
    int_1003 = 1003
    int_1004 = 1004


class AspectRatio(str, Enum):
    field_16_9 = '16:9'
    field_9_16 = '9:16'
    field_1_1 = '1:1'


class Config(BaseModel):
    horizontal: Optional[confloat(ge=-10.0, le=10.0)] = None
    pan: Optional[confloat(ge=-10.0, le=10.0)] = None
    roll: Optional[confloat(ge=-10.0, le=10.0)] = None
    tilt: Optional[confloat(ge=-10.0, le=10.0)] = None
    vertical: Optional[confloat(ge=-10.0, le=10.0)] = None
    zoom: Optional[confloat(ge=-10.0, le=10.0)] = None


class Type(str, Enum):
    simple = 'simple'
    down_back = 'down_back'
    forward_up = 'forward_up'
    right_turn_forward = 'right_turn_forward'
    left_turn_forward = 'left_turn_forward'


class CameraControl(BaseModel):
    config: Optional[Config] = None
    type: Optional[Type] = Field(None, description='Predefined camera movements type')


class Duration(str, Enum):
    field_5 = 5
    field_10 = 10


class Mode(str, Enum):
    std = 'std'
    pro = 'pro'


class TaskInfo(BaseModel):
    external_task_id: Optional[str] = None


class Video(BaseModel):
    duration: Optional[str] = Field(None, description='Total video duration')
    id: Optional[str] = Field(None, description='Generated video ID')
    url: Optional[AnyUrl] = Field(None, description='URL for generated video')


class TaskResult(BaseModel):
    videos: Optional[List[Video]] = None


class TaskStatus(str, Enum):
    submitted = 'submitted'
    processing = 'processing'
    succeed = 'succeed'
    failed = 'failed'


class Data(BaseModel):
    created_at: Optional[int] = Field(None, description='Task creation time')
    task_id: Optional[str] = Field(None, description='Task ID')
    task_info: Optional[TaskInfo] = None
    task_result: Optional[TaskResult] = None
    task_status: Optional[TaskStatus] = None
    updated_at: Optional[int] = Field(None, description='Task update time')


class AspectRatio1(str, Enum):
    field_16_9 = '16:9'
    field_9_16 = '9:16'
    field_1_1 = '1:1'
    field_4_3 = '4:3'
    field_3_4 = '3:4'
    field_3_2 = '3:2'
    field_2_3 = '2:3'
    field_21_9 = '21:9'


class ImageReference(str, Enum):
    subject = 'subject'
    face = 'face'


class Image(BaseModel):
    index: Optional[int] = Field(None, description='Image Number (0-9)')
    url: Optional[AnyUrl] = Field(None, description='URL for generated image')


class TaskResult1(BaseModel):
    images: Optional[List[Image]] = None


class Data1(BaseModel):
    created_at: Optional[int] = Field(None, description='Task creation time')
    task_id: Optional[str] = Field(None, description='Task ID')
    task_result: Optional[TaskResult1] = None
    task_status: Optional[TaskStatus] = None
    task_status_msg: Optional[str] = Field(None, description='Task status information')
    updated_at: Optional[int] = Field(None, description='Task update time')


class AspectRatio2(str, Enum):
    field_16_9 = '16:9'
    field_9_16 = '9:16'
    field_1_1 = '1:1'


class CameraControl1(BaseModel):
    config: Optional[Config] = None
    type: Optional[Type] = Field(None, description='Predefined camera movements type')


class ModelName2(str, Enum):
    kling_v1 = 'kling-v1'
    kling_v1_6 = 'kling-v1-6'


class TaskResult2(BaseModel):
    videos: Optional[List[Video]] = None


class Data2(BaseModel):
    created_at: Optional[int] = Field(None, description='Task creation time')
    task_id: Optional[str] = Field(None, description='Task ID')
    task_info: Optional[TaskInfo] = None
    task_result: Optional[TaskResult2] = None
    task_status: Optional[TaskStatus] = None
    updated_at: Optional[int] = Field(None, description='Task update time')


class Code2(Enum):
    int_1200 = 1200
    int_1201 = 1201
    int_1202 = 1202
    int_1203 = 1203


class ResourcePackType(str, Enum):
    decreasing_total = 'decreasing_total'
    constant_period = 'constant_period'


class Status(str, Enum):
    toBeOnline = 'toBeOnline'
    online = 'online'
    expired = 'expired'
    runOut = 'runOut'


class ResourcePackSubscribeInfo(BaseModel):
    effective_time: Optional[int] = Field(
        None, description='Effective time, Unix timestamp in ms'
    )
    invalid_time: Optional[int] = Field(
        None, description='Expiration time, Unix timestamp in ms'
    )
    purchase_time: Optional[int] = Field(
        None, description='Purchase time, Unix timestamp in ms'
    )
    remaining_quantity: Optional[float] = Field(
        None, description='Remaining quantity (updated with a 12-hour delay)'
    )
    resource_pack_id: Optional[str] = Field(None, description='Resource package ID')
    resource_pack_name: Optional[str] = Field(None, description='Resource package name')
    resource_pack_type: Optional[ResourcePackType] = Field(
        None,
        description='Resource package type (decreasing_total=decreasing total, constant_period=constant periodicity)',
    )
    status: Optional[Status] = Field(None, description='Resource Package Status')
    total_quantity: Optional[float] = Field(None, description='Total quantity')

class Background(str, Enum):
    transparent = 'transparent'
    opaque = 'opaque'


class Moderation(str, Enum):
    low = 'low'
    auto = 'auto'


class OutputFormat(str, Enum):
    png = 'png'
    webp = 'webp'
    jpeg = 'jpeg'


class Quality(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


class OpenAIImageEditRequest(BaseModel):
    background: Optional[str] = Field(
        None, description='Background transparency', examples=['opaque']
    )
    model: str = Field(
        ..., description='The model to use for image editing', examples=['gpt-image-1']
    )
    moderation: Optional[Moderation] = Field(
        None, description='Content moderation setting', examples=['auto']
    )
    n: Optional[int] = Field(
        None, description='The number of images to generate', examples=[1]
    )
    output_compression: Optional[int] = Field(
        None, description='Compression level for JPEG or WebP (0-100)', examples=[100]
    )
    output_format: Optional[OutputFormat] = Field(
        None, description='Format of the output image', examples=['png']
    )
    prompt: str = Field(
        ...,
        description='A text description of the desired edit',
        examples=['Give the rocketship rainbow coloring'],
    )
    quality: Optional[str] = Field(
        None, description='The quality of the edited image', examples=['low']
    )
    size: Optional[str] = Field(
        None, description='Size of the output image', examples=['1024x1024']
    )
    user: Optional[str] = Field(
        None,
        description='A unique identifier for end-user monitoring',
        examples=['user-1234'],
    )


class Quality1(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'
    standard = 'standard'
    hd = 'hd'


class ResponseFormat(str, Enum):
    url = 'url'
    b64_json = 'b64_json'


class Style(str, Enum):
    vivid = 'vivid'
    natural = 'natural'


class OpenAIImageGenerationRequest(BaseModel):
    background: Optional[Background] = Field(
        None, description='Background transparency', examples=['opaque']
    )
    model: Optional[str] = Field(
        None, description='The model to use for image generation', examples=['dall-e-3']
    )
    moderation: Optional[Moderation] = Field(
        None, description='Content moderation setting', examples=['auto']
    )
    n: Optional[int] = Field(
        None,
        description='The number of images to generate (1-10). Only 1 supported for dall-e-3.',
        examples=[1],
    )
    output_compression: Optional[int] = Field(
        None, description='Compression level for JPEG or WebP (0-100)', examples=[100]
    )
    output_format: Optional[OutputFormat] = Field(
        None, description='Format of the output image', examples=['png']
    )
    prompt: str = Field(
        ...,
        description='A text description of the desired image',
        examples=['Draw a rocket in front of a blackhole in deep space'],
    )
    quality: Optional[Quality1] = Field(
        None, description='The quality of the generated image', examples=['high']
    )
    response_format: Optional[ResponseFormat] = Field(
        None, description='Response format of image data', examples=['b64_json']
    )
    size: Optional[str] = Field(
        None,
        description='Size of the image (e.g., 1024x1024, 1536x1024, auto)',
        examples=['1024x1536'],
    )
    style: Optional[Style] = Field(
        None, description='Style of the image (only for dall-e-3)', examples=['vivid']
    )
    user: Optional[str] = Field(
        None,
        description='A unique identifier for end-user monitoring',
        examples=['user-1234'],
    )


class Datum1(BaseModel):
    b64_json: Optional[str] = Field(None, description='Base64 encoded image data')
    revised_prompt: Optional[str] = Field(None, description='Revised prompt')
    url: Optional[str] = Field(None, description='URL of the image')


class OpenAIImageGenerationResponse(BaseModel):
    data: Optional[List[Datum1]] = None
class User(BaseModel):
    email: Optional[str] = Field(None, description='The email address for this user.')
    id: Optional[str] = Field(None, description='The unique id for this user.')
    isAdmin: Optional[bool] = Field(
        None, description='Indicates if the user has admin privileges.'
    )
    isApproved: Optional[bool] = Field(
        None, description='Indicates if the user is approved.'
    )
    name: Optional[str] = Field(None, description='The name for this user.')