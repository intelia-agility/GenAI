import base64
import time
import typing
import math
import numpy as np

from google.cloud import aiplatform
from google.protobuf import struct_pb2

#libraries to generate image summaries
import vertexai
from vertexai.vision_models import Video
from vertexai.vision_models import VideoSegmentConfig
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part as GenerativeModelPart,
    HarmBlockThreshold,
    HarmCategory,
)
from typing import Any, Dict, List, Literal, Optional, Union

vertexai.init(project=285188988743, location="us-central1")

contents= [{
                                                                    "role": "user",
                                                                    "parts": [
                                                                        {
                                                                    
                                                                        
                                                                        "file_data": {
                                                                            "mime_type":  "video/mp4",
                                                                            "file_uri": "gs://nine_dry_run_showcase_assets/test/video/vlt_video_extract_OTHERS_same_video_with_sound_VIDEOS_MAFS_vlt_video_extract_MAAT_Full_MAAT2023_10_A_HBB - Trim with sound.mp4"
                                                                        } 
                                                                         ,
                                                                        "video_metadata": {
                                                                                        "start_offset": {
                                                                                        "seconds": 600,
                                                                                        "nanos": 0
                                                                                        },
                                                                                        "end_offset": {
                                                                                        "seconds": 900,
                                                                                        "nanos": 0
                                                                                        }
                                                                                }

                                                                        },
                                                                        { "text": "Describe this video for the given time period. Focus on key themes, what people are talking about, names of people who appeared and who are discussed and key locations. Give me at least 4000 words describing it."
                                                                         }
                                                                    ]
                                                                    }
                                                                ]

generative_multimodal_model= GenerativeModel("gemini-1.5-pro-002")

print('you are here')
safety_settings=  {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }

model_response = generative_multimodal_model.generate_content(
                                    contents ,safety_settings=safety_settings
                                   )   
print(model_response)