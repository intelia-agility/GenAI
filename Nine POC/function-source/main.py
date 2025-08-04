import functions_framework
import urllib.request
from google.cloud import storage
import copy
import numpy as np
import requests
from tenacity import retry, stop_after_attempt

import base64
import time
import typing
import copy
import requests
from typing import List, Optional

from google.cloud import aiplatform
from google.protobuf import struct_pb2

#libraries to generate image summaries
import vertexai
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
import tempfile, shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generator, List, Optional
from tqdm.auto import tqdm
import json
from datetime import datetime
from google.auth import default


#Number of API calls per second
API_IMAGES_PER_SECOND = 2
#Number of files to be processed in a batch
BATCH_SIZE = 1# this can be changed
#source bucket to get data from
SOURCE_BUCKET_NAME='raw_nine_files'
#MM embeddings folder name
MM_UNIQUE_FOLDER_NAME = "multimodal_embeddings"
#Content embeddings folder name
CNT_UNIQUE_FOLDER_NAME = "content_embeddings"
#Destination bucket name
DESTINATION_BUCKET_URI = f"artifacts-nine-quality-test-embeddings" 
PROJECT_ID = 'nine-quality-test'
REGION ='us-central1'

vertexai.init(project=PROJECT_ID, location=REGION)
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]
 
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

    def get_embedding(self, text: str = None, image_file: str = None):
        if not text and not image_file:
            raise ValueError("At least one of text or image_file must be specified.")


        instance = struct_pb2.Struct()
        if text:
            instance.fields["text"].string_value = text        
        
        if image_file:
            if (image_file.startswith("gs://")): 
                  instance["image"] = {
                        "gcsUri": image_file  # pylint: disable=protected-access
                    }       
        
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
        if image_file:
            image_emb_value = response.predictions[0]["imageEmbedding"]
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding, image_embedding=image_embedding
        )

    def get_image_summarycontent(self, text: str = None, image_file: str = None):
        
        """
        Generates summary content for the image.

        Args:
            image_file: The input image url to summarize its content.

        Returns:
            string: string value including the content description of the provided image.
        """            
        
        generative_multimodal_model= GenerativeModel("gemini-pro-vision")
        
        image_description_prompt="""You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval.\
        If there is a famous person like politician, celebrity or athlete, indicate their name and describe what they are famous for.\
        If you are not sure about any info, please do not make it up."""
        
        generation_config= GenerationConfig(temperature=0.2, max_output_tokens=2048) 
        
        safety_settings=  {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        stream=True
        
        # Load the saved image as a Gemini Image Object
        #image_for_gemini= Image.load_from_file(image_file)
        image_for_gemini = Part.from_uri(image_file, "image/jpeg")

        model_input=[image_description_prompt, image_for_gemini]
        
        response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=generation_config,
        stream=stream,
        safety_settings=safety_settings, )
        
        
        response_list = []

        for chunk in response:
            try:
                response_list.append(chunk.text)
            except Exception as e:
                print(
                    "Exception occurred while calling gemini. Something is wrong. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                    e,
                )
                response_list.append("Exception occurred")
                continue
        response = "".join(response_list)
 
        return response

    def get_summarycontent_embedding_from_text_embedding_model(self, text: str, return_array: Optional[bool] = False,) -> list:
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
    
    
def generate_batches(
    inputs: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    """
    Generator function that takes a list of strings and a batch size, and yields batches of the specified size.
    """

    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]



def encode_to_embeddings_chunked(
    process_function: Callable[[List[str]], List[Optional[List[float]]]],
    items: List[str],
    batch_size: int = 1,
) -> List[Optional[List[float]]]:
    """
    Function that encodes a list of strings into embeddings using a process function.
    It takes a list of strings and returns a list of optional lists of floats.
    The data is processed in chunks to prevent out-of-memory errors.
    """

    embeddings_list: List[Optional[List[float]]] = []

    # Prepare the batches using a generator
    batches = generate_batches(items, batch_size)

    seconds_per_job = batch_size / API_IMAGES_PER_SECOND

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(batches, total=len(items) // batch_size, position=0):
            futures.append(executor.submit(process_function, batch))
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())
    return embeddings_list

    
client = EmbeddingPredictionClient(project=PROJECT_ID)


# Use a retry handler in case of failure
@retry(reraise=True, stop=stop_after_attempt(3))
def encode_texts_to_embeddings_with_retry(text: List[str]) -> List[List[float]]:
    assert len(text) == 1

    try:
        return [client.get_embedding(text=text[0], image_file=None).text_embedding]
    except Exception:
        raise RuntimeError("Error getting embedding.")


def encode_texts_to_embeddings(text: List[str]) -> List[Optional[List[float]]]:
    try:
        return encode_texts_to_embeddings_with_retry(text=text)
    except Exception:
        return [None for _ in range(len(text))]


@retry(reraise=True, stop=stop_after_attempt(3))
def encode_images_to_embeddings_with_retry(image_uris: List[str]) -> List[List[float]]:
    assert len(image_uris) == 1

    try:
        return [
            client.get_embedding(text=None, image_file=image_uris[0]).image_embedding
        ]
    except Exception as ex:
        print(ex)
        raise RuntimeError("Error getting embedding for image.")


def encode_images_to_embeddings(image_uris: List[str]) -> List[Optional[List[float]]]:
    try:
        return encode_images_to_embeddings_with_retry(image_uris=image_uris)
    except Exception as ex:
        print(ex)
        return [None for _ in range(len(image_uris))]
    

@retry(reraise=True, stop=stop_after_attempt(3))
def encode_images_to_summarycontent_with_retry(image_uris: List[str]) -> List[List[float]]:
    assert len(image_uris) == 1

    try:
        return [
            client.get_image_summarycontent(text=None, image_file=image_uris[0])
        ]
    except Exception as ex:
        print(ex)
        raise RuntimeError("Error getting summaries.")


def encode_images_to_summarycontent(image_uris: List[str]) -> List[Optional[List[float]]]:
    try:
        return encode_images_to_summarycontent_with_retry(image_uris=image_uris)
    except Exception as ex:
        print(ex)
        return [None for _ in range(len(image_uris))]
    
    
# Use a retry handler in case of failure
@retry(reraise=True, stop=stop_after_attempt(3))
def encode_summarycontent_to_embeddings_with_retry(text: List[str]) -> List[List[float]]:
    assert len(text) == 1

    try:
        return [client.get_summarycontent_embedding_from_text_embedding_model(text=text[0]).text_embedding]
    except Exception:
        raise RuntimeError("Error getting embedding for summary content.")


def encode_summarycontent_to_embeddings(text: List[str]) -> List[Optional[List[float]]]:
    try:
        return encode_summarycontent_to_embeddings_with_retry(text=text)
    except Exception:
        return [None for _ in range(len(text))]
    
def list_blobs(bucket_name: str):
    """Lists all the blobs in the bucket. 
   
        Args:
            bucket_name: name of the source gcs bucket to read files from

        Returns:
            list paths: list of paths to the gcs objects
            list names: list of names of the gcs objects
    """
        
    # bucket_name = "your-bucket-name"
    paths=[]
    names=[]
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name,match_glob=["**/*.png"])
 
    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:    
        paths.append('gs://' + blob.id[:-(len(str(blob.generation)) + 1)])
        names.append(blob.name)
    return paths,names
   
def upload_embeddings_to_gcs(embeddings_file: tempfile, file_pre_fix:str, folder_name:str, dest_bucket_name:str ):
    temp=embeddings_file
    client = storage.Client()
    now=datetime.strptime(str(datetime.now()),
                               '%Y-%m-%d %H:%M:%S.%f')
    # Extract name to the temp file
    temp_file = "".join([str(temp.name)])
    # Uploading the temp image file to the bucket
    dest_filename = f"{folder_name}/"+file_pre_fix+datetime.strftime(now, '%Y%m%d%H%M%S')+".json" 
    dest_bucket = client.get_bucket(dest_bucket_name)
    dest_blob = dest_bucket.blob(dest_filename)
    dest_blob.upload_from_filename(temp_file)
        
# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def Generate_Batch_Image_Embeddings(cloud_event):
    
    #get the list of images from the bucket
    image_paths,image_names=list_blobs(SOURCE_BUCKET_NAME)
    
    # Create temporary file to write embeddings to
    embeddings_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    # Create temporary file to write summaries to
    summaries_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False) 

    with open(embeddings_file.name, "a") as ef, open(summaries_file.name, "a") as sf:
         for i in tqdm(range(0, len(image_names), BATCH_SIZE)):#len(image_names)
            image_names_chunk = image_names[i : i + BATCH_SIZE]
            image_paths_chunk = image_paths[i : i + BATCH_SIZE]
            embeddings=[]
            image_summaries=[]

            #********************************
            embeddings = encode_to_embeddings_chunked(
                process_function=encode_images_to_embeddings, items=image_paths_chunk
            )

            #********************************
            summaries = encode_to_embeddings_chunked(
                process_function=encode_images_to_summarycontent, items=image_paths_chunk
               )
 
            #********************************

            summaries_embeddings = encode_to_embeddings_chunked(
                 process_function=encode_summarycontent_to_embeddings, items=summaries
            )

            #********************************

            # Append to file
            embeddings_formatted = [
                json.dumps(
                    {
                        "id": str(id),
                        "embedding": [str(value) for value in embedding],
                    }
                )
                + "\n"
                for id, embedding in zip(image_names_chunk, embeddings)
                if embedding is not None
            ]
            ef.writelines(embeddings_formatted)


            summaries_formatted = [
                json.dumps(
                    {
                        "id": str(id),
                        "image path": image_path,
                        "summary":  summary,
                        "summary embedding": [str(value) for value in summaries_embedding],
                        "image embedding": [str(value) for value in embedding],
                    }
                )
                + "\n"
                for id, summary,summaries_embedding,embedding,image_path in zip(image_names_chunk, summaries,summaries_embeddings,embeddings,image_paths_chunk)
                if summaries is not None
            ]
            sf.writelines(summaries_formatted)

    #push files into destination buckets
    upload_embeddings_to_gcs(embeddings_file, 'multimodal', MM_UNIQUE_FOLDER_NAME, DESTINATION_BUCKET_URI )
    upload_embeddings_to_gcs(summaries_file, 'content', CNT_UNIQUE_FOLDER_NAME, DESTINATION_BUCKET_URI )
    print('Embeddings pushed to gcs')
    #to do : move the processed files into processed folder