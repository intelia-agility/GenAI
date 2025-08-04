import google.cloud.bigquery as bq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore,BigQueryLoader
from datetime import datetime
import os
import logging
import json
    
def chunk_and_load(project_id: str= None, dataset: str= None, table: str= None, region: str =None,\
                        metadata_columns: list[str]=None, page_content_columns: list[str]= None, \
                        source_query_str: str= None, separators:  list[str]=None, chunk_size: int=None, \
                       chunk_overlap: int=0):
    """
        Chunks the combination of page_content_columns from a given bigquery source string and generates
        text embeddings for them.

        Args:
            

        Returns:
             
        """


     # Load the data
    loader = BigQueryLoader(
        source_query_str, metadata_columns=metadata_columns, page_content_columns=page_content_columns
    )

    documents = []
    documents.extend(loader.load())
    
    logging.info (f"Data Loaded from source - {source_query_str}")
   
    print('you are here')
    print(chunk_size)
    print(chunk_overlap)

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
    logging.info (f"Metadata added to chunks")

    from google.cloud import bigquery

    client = bigquery.Client()
    table_id = f"{project_id}.{dataset}.{table}"

 
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print(f"Encountered errors: {errors}")
    
    
    
#     logging.info (f"Bigquery store info is set -  ProjectID {project_id}, Region {region}, Dataset {dataset}, Table {table}")


#     _=bq_store.add_documents(doc_splits)
    
#     logging.info (f"Chunks and embeddings added to the store")

    return 'done'

if __name__ == "__main__":
    project_id= os.environ.get("PROJECT_ID") 
    dataset= os.environ.get("DATASET")  
    table= os.environ.get("TABLE") 
    region= os.environ.get("REGION") 
    metadata_columns= str(os.environ.get("META_DATA_COLUMNS")).split(',') 
    page_content_columns= str(os.environ.get("PAGE_CONTENT_COLUMNS")).split(',') 
    source_query_str= os.environ.get("SOURCE_QUERY_STR") 
    separators= None if str(os.environ.get("SEPARATORS"))=="" else str(os.environ.get("SEPARATORS")).split(',') 
    chunk_size= 1000 if str(os.environ.get("CHUNK_SIZE"))=="None" else int(str(os.environ.get("CHUNK_SIZE")))  
    chunk_overlap= 0 if str(os.environ.get("CHUNK_OVERLAP"))=="None" else int(str(os.environ.get("CHUNK_OVERLAP")))
 
 
    print(chunk_size)
    print(chunk_overlap)

    # project_id= 'nine-quality-test' 
    # dataset= 'my_langchain_dataset'
    # table= 'doc_and_vectors'
    # region= 'us-central1'
    # metadata_columns= "id".split(",")
    # page_content_columns= "content,media_type,test_metadata".split(',') 
    # source_query_str= """
    # SELECT id,media_type,content,test_metadata 
    # FROM `nine-quality-test.Nine_Quality_Test.content_embeddings` ;
    # """
    # separators= "\n\n"
    # chunk_size= 25
    # chunk_overlap= 5
    
    
    message=chunk_and_load(project_id=project_id, dataset=dataset, table=table, region=region,\
                        metadata_columns= metadata_columns, page_content_columns=page_content_columns, \
                        source_query_str=source_query_str, separators=separators, chunk_size=chunk_size, \
                       chunk_overlap=chunk_overlap)
    print(message)
     