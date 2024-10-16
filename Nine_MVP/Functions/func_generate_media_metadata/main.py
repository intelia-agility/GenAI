import functions_framework
from datetime import datetime
import logging
from google.cloud import bigquery
import os, json
from google.cloud import storage


def create_dataset(project_id,dataset_id,region_id):
    """
        Create a dataset
        
        Args:
           str project_id: project id
           str dataset_id: name of the dataset under which the table should be created.
           str region_id:  name of the region under whcih the dataset should be created.               
    """
    
    client = bigquery.Client(project_id)
    print(project_id)
    # Check if the dataset exists
    try:
        dataset = client.get_dataset(f"{project_id}.{dataset_id}")  # Make an API request.
        print(f"Dataset '{dataset_id}' already exists.")
    except :
        # If the dataset does not exist, create it
        
        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset.location = region_id # Set your desired location

        # Create the dataset
        dataset = client.create_dataset(dataset)  # Make an API request.
        print(f"Dataset '{dataset_id}' created successfully.")
        
 
def create_table(project_id,dataset_id,table_id):
    
    """
        Create a table for video/image metadata
        
        Args:
           str project_id: project id
           str dataset_id: name of the dataset under which the table should be created.
           str table_id:  name of the table.
        Returns:
            list[bigquery.SchemaField] schema: table schema- list of columns
             
    """
        
    
    client = bigquery.Client(project_id)
    
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # Define the table schema
    schema = [
        bigquery.SchemaField("idx", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("mime_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("gcs_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("media_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("predictions", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("time", "STRING", mode="NULLABLE")
       
    ]     

    # Check if the table exists
    try:
        table = client.get_table(full_table_id)  # Make an API request.
        print(f"Table '{full_table_id}' already exists.")
        
        # Drop the table if exist
        query = f"DROP TABLE `{full_table_id}`"

        # Execute the query
        try:
            client.query(query).result()  # Make an API request.
            print(f"Table '{full_table_id}' dropped successfully.")
        except Exception as e:
            print(f"Error dropping table '{full_table_id}': {e}")
    except :
        # If the table does not exist, create it
        table = bigquery.Table(full_table_id, schema=schema)

        # Create the table
        table = client.create_table(table)  # Make an API request.
        print(f"Table '{full_table_id}' created successfully.")
        
    return schema

def create_video_landing(project_id,dataset_id,table_id):
    
    """
        Create a table for video landing
        
        Args:
           str project_id: project id
           str dataset_id: name of the dataset under which the table should be created.
           str table_id:  name of the table.
        Returns:
            list[bigquery.SchemaField] schema: table schema- list of columns
             
    """
        
    
    client = bigquery.Client(project_id)
    
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # Define the table schema
    schema = [
     
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("mime_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("gcs_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("media_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("predictions", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("start_offset", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("end_offset", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("time", "STRING", mode="NULLABLE")
    ]     

    # Check if the table exists
    try:
        table = client.get_table(full_table_id)  # Make an API request.
        print(f"Table '{full_table_id}' already exists.")
        
        # Drop the table if exist
        query = f"DROP TABLE `{full_table_id}`"

        # Execute the query
        try:
            client.query(query).result()  # Make an API request.
            print(f"Table '{full_table_id}' dropped successfully.")
        except Exception as e:
            print(f"Error dropping table '{full_table_id}': {e}")
    except :
        # If the table does not exist, create it
        table = bigquery.Table(full_table_id, schema=schema)

        # Create the table
        table = client.create_table(table)  # Make an API request.
        print(f"Table '{full_table_id}' created successfully.")
        
    return schema

@functions_framework.http
def get_media_metadata(request):
    """
        loads gcs metadata info into big query table
             
    """
    request_args = request.args 

    # project_id=  request_args['project_id']
    # dataset_id=  request_args['dataset']
    # table= request_args['table']
    # region= request_args['region']
    # source_bucket= request_args['source_bucket']
    # source_folder= request_args['source_folder']
    # media_types= [media.strip() for media in  str(request_args['media_types']).strip().replace("[",''). replace(']','').replace("'",'').split(',')]
    project_id=  request_args['project_id']
    dataset_id=  request_args['dataset']
    table= request_args['table']
    region= request_args['region']
    source_bucket= request_args['source_bucket']
    source_folder= request_args['source_folder']

    media_types= [media.strip() for media in  str(request_args['media_types']).strip().replace("[",''). replace(']','').replace("'",'').split(',')]
  

    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.get_bucket(source_bucket)
    # List all objects in the bucket
    if source_folder=="":
        blobs = bucket.list_blobs()    
    else:
        blobs = bucket.list_blobs(prefix=source_folder)
    
    # get current processing time to add it to metadata, datetime object containing current date and time
    now = datetime.now()  
    rows_to_insert=[]        
    client = bigquery.Client(project_id)    
   
    #create data set if does not exist
    create_dataset(project_id,dataset_id,region)
    
    #create video landingtable
    if request_args and 'video_landing_table' in request_args:
        #we are processing videos; create a landing for videos
        video_landing_table= request_args['video_landing_table']
        if video_landing_table.strip()!="" and  video_landing_table.strip().lower()!="none":
             _=create_video_landing(project_id,dataset_id,video_landing_table)
     


    job_list=[]
    idx=0
    for blob in blobs:
        if blob.content_type in media_types:
            rows_to_insert.append(
                                {   "idx":  idx  , 
                                    "name": blob.name, 
                                    "mime_type":blob.content_type,
                                    "gcs_uri":  "gs://"+source_bucket+"/"+blob.name,
                                    "media_name":blob.name.split("/")[-1].replace("."+blob.content_type.split("/")[-1],"")
                                    }
                                            )
        
            idx=idx+1

    #create table new if does not exist
    table=f"{table}" 
    table_schema=create_table(project_id,dataset_id,table)
    #_=create_error_table(project_id,dataset_id,error_table)
    #push the data into the table
    table_id = f"{project_id}.{dataset_id}.{table}"
    dataset  = client.dataset(dataset_id)
    table = dataset.table(table)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.schema = table_schema
    job = client.load_table_from_json(rows_to_insert, table, job_config = job_config)             
    job_list.append(job.job_id)
  
    return {'status':'SUCCESS', 'record_count':idx,'count_of_tables':len(job_list),'jobs':job_list }