import functions_framework
from flask import  jsonify, request
from google.cloud import storage
import tempfile, shutil
import time
from typing import Callable, Generator, List, Optional 
from datetime import datetime
import json
 
def create_request_file(requests: list, file_pre_fix: str, dest_bucket_name: str, dest_folder_name: str):
    """create batch request file and store it in gcs
   
        Args:
            str dest_bucket_name: name of the destination gcs bucket to write files to
            str dest_folder_name: name of the destination folder name to write files to
            str file_pre_fix: prefix of the destination file name
            list requests: list of requests

        Returns:
            str path:  path to the gcs request file object
          
    """
        
      # Create temporary file to write summaries to
    request_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False) 
 
    with open(request_file.name, "a") as rf :
         for request in requests:
             request_list = [json.dumps(request)+  "\n"]
             rf.writelines(request_list)

    temp=request_file
    client = storage.Client()
    now=datetime.strptime(str(datetime.now()),
                               '%Y-%m-%d %H:%M:%S.%f')
    # Extract name to the temp file
    temp_file = "".join([str(temp.name)])
    # Uploading the temp image file to the bucket
    dest_filename = f"{dest_folder_name}/"+file_pre_fix+datetime.strftime(now, '%Y%m%d%H%M%S')+".json" 
    dest_bucket = client.get_bucket(dest_bucket_name)
    dest_blob = dest_bucket.blob(dest_filename)
    dest_blob.upload_from_filename(temp_file)

    return dest_filename

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

    bucket = request_args['bucket']
    request_file_prefix = request_args['request_file_prefix']
    request_file_folder = request_args['request_file_folder']
    request_body= request_args['request_body']
   
    requests_list=  json.loads(request_body)  
    dest_filename=create_request_file(requests_list,request_file_prefix,bucket,request_file_folder)  

    return {"request_file_name":dest_filename}
