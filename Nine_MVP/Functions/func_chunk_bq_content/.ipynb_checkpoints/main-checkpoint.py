import functions_framework

import google.cloud.bigquery as bq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_community import BigQueryLoader
from datetime import datetime
import logging
from google.cloud import bigquery
import os, json

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
        Create a table 
        
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
        bigquery.SchemaField("request_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("original_content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("media_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("path", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("test_metadata", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("chunk", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("process_time", "DATETIME", mode="REQUIRED"),
    ]
                

    # Check if the table exists
    try:
        table = client.get_table(full_table_id)  # Make an API request.
        print(f"Table '{full_table_id}' already exists.")
        
        # Truncate the table if exist
        query = f"TRUNCATE TABLE `{full_table_id}`"

        # Execute the query
        try:
            client.query(query).result()  # Make an API request.
            print(f"Table '{full_table_id}' truncated successfully.")
        except Exception as e:
            print(f"Error truncating table '{full_table_id}': {e}")
    except :
        # If the table does not exist, create it
        table = bigquery.Table(full_table_id, schema=schema)

        # Create the table
        table = client.create_table(table)  # Make an API request.
        print(f"Table '{full_table_id}' created successfully.")
        
    return schema


@functions_framework.http
def chunk_bq_content(request):
    """
        Chunks the combination of page_content_columns from a given bigquery source string and load the result into bigquery
             
    """
    status=''  
    request_args = request.args
 

    project_id=  request_args['project_id']
    dataset_id=  request_args['dataset']
    table= request_args['table']
    region= request_args['region']
    metadata_columns= [col.strip() for col in  str(request_args['metadata_columns']).split(',') ]
    page_content_columns= [col.strip() for col in str(request_args['page_content_columns']).split(',') ]
    source_query_str= request_args['source_query_str']
    #separators= "\n" if str(request_args['separators'])=="" else str(request_args['separators']).split(',') 
    chunk_size= 1000 if str(request_args['chunk_size']) in ["None",""] else int(str(request_args['chunk_size']))  
    chunk_overlap= 0 if str(request_args['chunk_overlap']) in ["None",""] else int(str(request_args['chunk_overlap']))  
    # # except Exception as e: 
    # if 1==1:
    #         # project_id= 'nine-quality-test' 
    #         # dataset_id= 'langchain_dataset'
    #         # table= 'chunked_data'
    #         # region= 'us-central1'
    #         metadata_columns= [ col.strip() for col in  "id, content, path, media_type, test_metadata".split(",")]
    #         page_content_columns= [col.strip() for col in "content, test_metadata".split(',') ]
    #         source_query_str= """
    #         SELECT id,media_type,content,test_metadata, path
    #         FROM `nine-quality-test.Nine_Quality_Test.test_long_artcles` ;
    #         """
    #         separators= "\n"
    #         chunk_size= 500
    #         chunk_overlap= 10
     

            #return {'record_count':0, 'status':'ERROR- Set required input parameters '+e}



     # Load the data
    loader = BigQueryLoader(
        query=source_query_str, project=project_id, metadata_columns=metadata_columns, page_content_columns=page_content_columns
    )
    
    documents = []
    documents.extend(loader.load())
    
    logging.info (f"Data Loaded from source - {source_query_str}")


    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
       #separators=separators,
    )
    doc_splits = text_splitter.split_documents(documents)    
     
    logging.info (f"Documents splitted - chunk_size {chunk_size}, chunk_overlap {chunk_overlap}")

    # get current processing time to add it to metadata, datetime object containing current date and time
    now = datetime.now()
    # Add chunk number to metadata
    chunk_idx=0
    prev=doc_splits[0].metadata["id"]
    rows_to_insert=[]
    request_date=datetime.today().strftime('%Y_%m_%d')    
        
    client = bigquery.Client(project_id)    
    #create data set if does not exist
    create_dataset(project_id,dataset_id,region)
    max_index=30000 #maximum number of requests in a batch
 
    max_index=2
    record_count=0
    prefix=f"{table}_{request_date}" 
    for idx, split in enumerate(doc_splits):
            split.metadata["process_time"]=now
            if prev==split.metadata["id"]:
               split.metadata["chunk"] = chunk_idx      
            else:
                chunk_idx=0
                split.metadata["chunk"] = chunk_idx
                prev=split.metadata["id"]
                
            chunk_idx +=1
            version=idx // max_index
            request_id = request_date+'_'+str(version)
            rows_to_insert.append(
                               {  "request_id":  request_id  , 
                                   "id": split.metadata["id"], 
                                   "process_time":split.metadata["process_time"].isoformat(),
                                   "content":  split.page_content,
                                   "original_content": split.metadata["content"],
                                   "chunk": split.metadata["chunk"],
                                   "media_type": split.metadata["media_type"],
                                   "path": split.metadata["path"],
                                   "test_metadata": split.metadata["test_metadata"]                            

                                  }
                                         )
            
            if (idx+1) % max_index==0:
               
                #create table new if does not exist
                table=f"{table}_{request_id}"
                table_schema=create_table(project_id,dataset_id,table)
                #push the data into the table
                table_id = f"{project_id}.{dataset_id}.{table}"
                dataset  = client.dataset(dataset_id)
                table = dataset.table(table)
                job_config = bigquery.LoadJobConfig()
                job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
                job_config.schema = table_schema
                job = client.load_table_from_json(rows_to_insert, table, job_config = job_config)
                #moving to next batch
                record_count=record_count+len(rows_to_insert)
                rows_to_insert=[]
    
    return {'status':'SUCCESS', 'record_count':record_count, 'count_of_tables':version+1, 'table_name_prefix':prefix}