import base64
import time
import typing
from typing import Optional
from google.cloud import aiplatform
from google.protobuf import struct_pb2

#libraries to generate image summaries
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
    HarmBlockThreshold,
    HarmCategory,
)

text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
#multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
 #   "multimodalembedding@001"
#)


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]

import requests

class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com",
    ):
        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        self.location = location
        self.project = project

    def get_multimodal_embedding(self, text: str = None, image_file: str = None):
        if not text and not image_file:
            raise ValueError("At least one of text or image_file must be specified.")

        # Load image file
        image_bytes = None
        if image_file:
            image_bytes = load_image_bytes(image_file)

        instance = struct_pb2.Struct()
        if text:
            instance.fields["text"].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields["image"].struct_value
            image_struct.fields["bytesBase64Encoded"].string_value = encoded_content

        instances = [instance]
        endpoint = (
           f"projects/{self.project}/locations/{self.location}"
           "/publishers/google/models/multimodalembedding@001"
        )
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]["textEmbedding"]
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]["imageEmbedding"]
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding, image_embedding=image_embedding
        ) 

    def get_text_embedding(self, text: str, return_array: Optional[bool] = False,) -> list:
        """
        Generates a numerical text embedding from a provided text input using a text embedding model.

        Args:
            text: The input text string to be embedded.
            return_array: If True, returns the embedding as a NumPy array.
                          If False, returns the embedding as a list. (Default: False)

        Returns:
            list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                                   The format (list or NumPy array) depends on the
                                   value of the 'return_array' parameter.
        """

        #the given text is maximum 2048 token. If more, it has to be chunked.
        embeddings = text_embedding_model.get_embeddings([text])
        text_embedding = [embedding.values for embedding in embeddings][0]

        if return_array:
            text_embedding = np.fromiter(text_embedding, dtype=float)

        # returns 768 dimensional array
        return EmbeddingResponse(
            text_embedding=text_embedding, image_embedding=None
        )
    