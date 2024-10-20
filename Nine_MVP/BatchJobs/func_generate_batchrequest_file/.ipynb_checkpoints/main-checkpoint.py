import functions_framework
from google.cloud import storage
import tempfile, shutil
import time
from datetime import datetime
import json
from google.cloud import bigquery 


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

    

def get_video_duration(gcsuri: str):
    """"
    get video duration of a given gcs url
    
    """"
    
    fs = gcsfs.GCSFileSystem()
    # Open the file stream using gcsfs
    with fs.open(gcsuri, 'rb') as video_file:
              # Use pymediainfo to extract metadata directly from the stream
              media_info = MediaInfo.parse(video_file)
              for track in media_info.tracks:
                  if track.track_type == 'Video':
                      duration= track.duration / 1000  # Convert ms to seconds
                      print(duration)
                      break
 

    return duration

def create_video_request_file( dest_bucket_name: str= None, source_bucket_name: str= None, source_folder_name: str=None,
                          request_file_prefix: str= None, request_file_folder: str= None, mime_types: list[str]= None,
                          prompt_text: str="", temperature: float= None,
                          max_output_tokens: int=2048, top_p: float= None, top_k : float= None, max_request_per_file: int =None, 
                          video_metadata_table: str="", intervals: int =120):

    """create batch request  file(s) of up to 30000 for video and store it in gcs
   
        Args:
            str dest_bucket_name: name of the destination gcs bucket to write files to
            str source_bucket_name: name of the source  gcs bucket to read files from
            str dest_folder_name: name of the destination folder name to write files to
            str source_folder_name: name of the source folder name to read files from
            str request_file_prefix: prefix of the request file name
            list mime_types: list of accepted mime_types
            str prompt_text: prompt for Gimini
            float temperature: Gimini temprature
            float top_p: Gimini top_p
            float top_k: Gimini top_k
            int max_request_per_file: max number of requests per batch
            int max_output_tokens: Gimini max_output_tokens          
            str video_metadata_table: name of video metadata table

        Returns:
            int : number of generated files
          
    """       

    # Initialize a client for interacting with GCS
    storage_client = storage.Client()
    # Get the bucket by name
    bucket = storage_client.get_bucket(source_bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs(prefix=source_folder_name)  
    version=0
    index=0     
    max_index=max_request_per_file

    now=datetime.strptime(str(datetime.now()),
                               '%Y-%m-%d %H:%M:%S.%f')
    now=datetime.strftime(now, '%Y%m%d%H%M%S')
    versions=[]   
    # Initialize the BigQuery client
    bq_client = bigquery.Client()

    segments_to_process=int(intervals) #segments duration  
    video_start =0 #where from video to start

    for blob in blobs:                         
        if blob.content_type in mime_types:                            
                         gcsuri= "gs://"+source_bucket_name+"/"+blob.name
                         mimeType=blob.content_type
 
                        video_duration=get_video_duration(gcsuri)
                                                
                         prev=video_start
                         for val in range (segments_to_process,video_duration+segments_to_process,segments_to_process):
                                offset={'start':prev, 'end':val}
                                
                                startOffset=offset['start']
                                endOffset=offset['end']
                                if endOffset>=video_duration:
                                     endOffset=video_duration
                                 
                                prev=val 
                                segment_prompt= "Describe this video from period " + str(startOffset)+" seconds to "+ str(endOffset)+" seconds." 
                                if index==0:
                                    request_file = tempfile.NamedTemporaryFile(suffix=".json", delete=True) 
                                    rf= open(request_file.name, "a") 
                                
                                 
                                request_list=[
                                       json.dumps(
                                              {
                                                "request": 
                                                     {
                                                      #"cached_content": cache_id,
                                                      "contents":  {
                                                                        "parts": [
                                                                            {
                                                                            "fileData":  {"fileUri": gcsuri, "mimeType": mimeType},
                                                                            "videoMetadata": {
                                                                                "endOffset": {
                                                                                "nanos": 0,
                                                                                "seconds": endOffset
                                                                                },
                                                                                "startOffset": {
                                                                                "nanos": 0,
                                                                                "seconds": startOffset
                                                                                }
                                                                            }

                                                                            },
                                                                            {
                                                                            "text": segment_prompt +"\n"+ prompt_text 
                                                                            } 

                                                                        ],
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

def create_image_request_file( dest_bucket_name: str= None, source_bucket_name: str= None, source_folder_name: str=None,
                          request_file_prefix: str= None, request_file_folder: str= None, mime_types: list[str]= None,
                          prompt_text: str="", temperature: float= None,
                          max_output_tokens: int=2048, top_p: float= None, top_k : float= None, max_request_per_file: int =None):

    """create batch request  file(s) of up to 30000 for videos and store it in gcs
   
        Args:
            str dest_bucket_name: name of the destination gcs bucket to write files to
            str source_bucket_name: name of the source  gcs bucket to read files from
            str dest_folder_name: name of the destination folder name to write files to
            str source_folder_name: name of the source folder name to read files from
            str request_file_prefix: prefix of the request file name
            list mime_types: list of accepted mime_types
            str prompt_text: prompt for Gimini
            float temperature: Gimini temprature
            float top_p: Gimini top_p
            float top_k: Gimini top_k
            int max_request_per_file: max number of requests per batch
            int max_output_tokens: Gimini max_output_tokens          

        Returns:
            int : number of generated files
          
    """           
         

    # Initialize a client for interacting with GCS
    storage_client = storage.Client()
    # Get the bucket by name
    bucket = storage_client.get_bucket(source_bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs(prefix=source_folder_name)  
    version=0
    index=0     
    max_index=max_request_per_file

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


        if request_args and 'intervals' in request_args:
            intervals= int(request_args['intervals'])
        else:
          intervals=120


        if request_args and 'video_metadata_table' in request_args:
            video_metadata_table= request_args['video_metadata_table']
        else:
          video_metadata_table=""

        if request_args and 'temperature' in request_args:
            temperature= float(request_args['temperature'])
        else:
          temperature=1

        if request_args and 'max_output_tokens' in request_args:
           max_output_tokens= int(request_args['max_output_tokens'] )
        else:
             max_output_tokens=8192

        if request_args and 'top_p' in request_args:
            top_p= float(request_args['top_p'])
        else:
             top_p=0.95

        if request_args and 'top_k' in request_args:
            top_k= float(request_args['top_k'])
        else:
             top_k=40

        if request_args and 'max_request_per_file' in request_args:
            max_request_per_file= int(request_args['max_request_per_file'])
        else:
          max_request_per_file=30000


        versions=0
        if  request_content=='image':
          versions=create_image_request_file(dest_bucket_name=dest_bucket_name,source_bucket_name=source_bucket_name,source_folder_name=source_folder_name,
                                          request_file_prefix=request_file_prefix,request_file_folder=request_file_folder,
                                          mime_types=media_types, prompt_text=prompt_text,temperature=temperature,
                                         max_output_tokens=max_output_tokens,top_p=top_p,top_k=top_k, max_request_per_file=max_request_per_file
                                          )  
        if  request_content=='video':
          versions=create_video_request_file(dest_bucket_name=dest_bucket_name,source_bucket_name=source_bucket_name,source_folder_name=source_folder_name,
                                          request_file_prefix=request_file_prefix,request_file_folder=request_file_folder,
                                          mime_types=media_types, prompt_text=prompt_text,temperature=temperature,
                                         max_output_tokens=max_output_tokens,top_p=top_p,top_k=top_k, max_request_per_file=max_request_per_file,
                                         video_metadata_table=video_metadata_table, intervals=intervals
                                          ) 

        return {"status":"SUCCESS","file_count":versions}
