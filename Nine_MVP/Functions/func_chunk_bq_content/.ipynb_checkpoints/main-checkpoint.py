import functions_framework
import google.cloud.bigquery as bq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_community import BigQueryVectorStore,BigQueryLoader
from datetime import datetime
import os
import logging
import json
from google.cloud import bigquery


@functions_framework.http
def chunk_bq_content(request):
    """
        Chunks the combination of page_content_columns from a given bigquery source string and generates
        text embeddings for them.
             
    """
    status=''
    request_args = request.args
    try:
        project_id=  request_args['project_id']
        dataset=  request_args['dataset']
        table= request_args['table']
        region= request_args['region']
        metadata_columns= str(request_args['metadata_columns']).split(',') 
        page_content_columns= str(request_args['page_content_columns']).split(',') 
        source_query_str= request_args['source_query_str']
        separators= None if str(request_args['separators'])=="" else str(request_args['separators']).split(',') 
        chunk_size= 1000 if str(request_args['chunk_size']) in ["None",""] else int(str(request_args['chunk_size']))  
        chunk_overlap= 0 if str(request_args['chunk_overlap']) in ["None",""] else int(str(request_args['chunk_overlap']))  
    except:
        return {'record_count':0, 'status':'ERROR- Set required input parameters'}



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
       separators=separators,
    )
    doc_splits = text_splitter.split_documents(documents)
    
   
    logging.info (f"Documents splitted - chunk_size {chunk_size}, chunk_overlap {chunk_overlap}")


    # get current processing time to add it to metadata, datetime object containing current date and time
    now = datetime.now()

    # Add chunk number to metadata
    chunk_idx=0
    prev=doc_splits[0].metadata["id"]
    rows_to_insert=[]
    for idx, split in enumerate(doc_splits):
        split.metadata["process_time"]=now
        if prev==split.metadata["id"]:
           split.metadata["chunk"] = chunk_idx      
        else:
            chunk_idx=0
            split.metadata["chunk"] = chunk_idx
            prev=split.metadata["id"]
        chunk_idx +=1

        rows_to_insert.append(
                           {"id": split.metadata["id"], 
                               "process_time":split.metadata["process_time"].isoformat(),
                               "content":  split.page_content,
                               "chunk": split.metadata["chunk"]
                              }
                                     )

    
    client = bigquery.Client(project_id)
    table_id = f"{project_id}.{dataset}.{table}"

 
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
        status='SUCCESS'
    else:
        print(f"Encountered errors: {errors}")
        status='ERROR- check execution logs'
    
    
    return {'record_count':len(rows_to_insert),'status':'SUCCESS'}