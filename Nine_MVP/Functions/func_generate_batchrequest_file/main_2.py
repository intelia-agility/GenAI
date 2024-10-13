import functions_framework
from google.cloud import storage
import tempfile, shutil
import time
from datetime import datetime
import json


def upload_file(request_file : tempfile,dest_bucket_name:str =None,request_file_folder: str =None,request_file_prefix: str =None, version: int=0, request_file_post_fix : str=""):

    """upload file into gcs
   
        Args:
            tempfile request_file: request file
            str dest_bucket_name:  name of destination bucket
            str request_file_folder: name of the destination folder name to write files to
            list request_file_prefix: prefix of request file name
          
    """

    temp=request_file
    client = storage.Client()
    # Extract name to the temp file
    temp_file = "".join([str(temp.name)])
    # Uploading the temp image file to the bucket
    dest_filename = f"{request_file_folder}/"+request_file_prefix+'_'+request_file_post_fix+'_'+str(version)+".json" 
    dest_bucket = client.get_bucket(dest_bucket_name)
    dest_blob = dest_bucket.blob(dest_filename)
    dest_blob.upload_from_filename(temp_file)                              


def create_image_request_file( dest_bucket_name: str= None, source_bucket_name: str= None, source_folder_name: str=None,
                          request_file_prefix: str= None, request_file_folder: str= None, mime_types: list[str]= None,
                          prompt_text: str="", temperature: float= None,
                          max_output_tokens: int=2048, top_p: float= None, top_k : float= None):

    """create batch request file(s) of up to 30000 and store it in gcs
   
        Args:
            str dest_bucket_name: name of the destination gcs bucket to write files to
            str dest_folder_name: name of the destination folder name to write files to
            str file_pre_fix: prefix of the destination file name
            list requests: list of requests

        Returns:
            int 1: as SUCCESS
          
    """       

    # Initialize a client for interacting with GCS
    storage_client = storage.Client()
    # Get the bucket by name
    bucket = storage_client.get_bucket(source_bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs(prefix=source_folder_name)  
    version=0
    index=0     
    max_index=30000

    now=datetime.strptime(str(datetime.now()),
                               '%Y-%m-%d %H:%M:%S.%f')
    now=datetime.strftime(now, '%Y%m%d%H%M%S')
    versions=[]

    for blob in blobs:                         
                    if blob.content_type in mime_types:                            
                         gcsuri= "gs://"+source_bucket_name+"/"+blob.name
                         mimeType=blob.content_type
                         if index==0:
                            request_file = tempfile.NamedTemporaryFile(suffix=".json", delete=True) 
                            rf= open(request_file.name, "a") 
                 
                         request_list=[
                               json.dumps(
                                      {
                                        "request": 
                                             {
                                              "contents": 
                                                  {"parts": [{ "fileData": 
                                                                 {"fileUri": gcsuri, "mimeType": mimeType}
                                                              }, 
                                                              {"text": prompt_text
                                                              }
                                                            ]
                                                              , 
                                                    "role": "user"
                                                  }
                                                  , 
                                                  "generation_config": 
                                                       {"max_output_tokens": max_output_tokens, 
                                                        "temperature":temperature, 
                                                         "top_k": top_k, 
                                                         "top_p": top_p
                                                        }
                                                  , 
                                                  "safety_settings": 
                                                   [
                                                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                                                     "threshold": "BLOCK_NONE"
                                                     }
                                                   , 
                                                   {"category": "HARM_CATEGORY_HATE_SPEECH", 
                                                   "threshold": "BLOCK_NONE"
                                                   },
                                                   {"category": "HARM_CATEGORY_HARASSMENT", 
                                                   "threshold": "BLOCK_NONE"
                                                   }, 
                                                   {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                                    "threshold": "BLOCK_NONE"
                                                    }


                                                   ]
                                             }
                                      }
                               )  +"\n"
                         ]

                         rf.writelines(request_list)
                         rf.flush()
 
                         if index==(max_index-1):
                                upload_file(rf,dest_bucket_name=dest_bucket_name,request_file_folder=request_file_folder,request_file_prefix=request_file_prefix,version=version, request_file_post_fix=now)
                                rf.close() 
                                versions.append(version)                               
                                index=0
                                version +=1
                                request_list=[]
                                rf=None

                         else:                               
                                index =index+1

    if not rf is None: 
        upload_file(rf,dest_bucket_name=dest_bucket_name,request_file_folder=request_file_folder,request_file_prefix=request_file_prefix,version=version,request_file_post_fix=now)
        versions.append(version)   
 
    return len(versions)

@functions_framework.http
def create_batch_request_file(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    dest_bucket_name =request_args['destination_bucket']
    source_bucket_name =request_args['source_bucket']
    source_folder_name=request_args['source_folder']
    request_file_prefix =request_args['request_file_prefix']
    request_file_folder =request_args['request_file_folder']
    prompt_text= request_args['prompt_text']
    media_types= [media.strip() for media in  str(request_args['media_types']).strip().replace("[",''). replace(']','').replace("'",'').split(',')]

 
    request_content= request_args['request_content']

    if request_args and 'temperature' in request_args:
        temperature= request_args['temperature']
    else:
      temperature=0.5

    if request_args and 'max_output_tokens' in request_args:
       max_output_tokens= request_args['max_output_tokens'] 
    else:
         max_output_tokens=2048

    if request_args and 'top_p' in request_args:
        top_p= request_args['top_p']
    else:
         top_p=0.5

    if request_args and 'top_k' in request_args:
        top_k= request_args['top_k']
    else:
         top_k=50

    versions=0
    if  request_content=='image':
      versions=create_image_request_file(dest_bucket_name=dest_bucket_name,source_bucket_name=source_bucket_name,source_folder_name=source_folder_name,
                                      request_file_prefix=request_file_prefix,request_file_folder=request_file_folder,
                                      mime_types=media_types, prompt_text=prompt_text,temperature=temperature,
                                     max_output_tokens=max_output_tokens,top_p=top_p,top_k=top_k
                                      )  

    return {"status":"SUCCESS","file_count":versions}
