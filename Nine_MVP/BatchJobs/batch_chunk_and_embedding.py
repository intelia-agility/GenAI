import google.cloud.bigquery as bq
from langchain_community.document_loaders import BigQueryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from datetime import datetime
from google.auth import default


def chunk_and_embedding():
    # Define our query
    query = """
    SELECT id,media_type,content,test_metadata 
    FROM `nine-quality-test.Nine_Quality_Test.content_embeddings` ;
    """

    # Load the data
    loader = BigQueryLoader(
        query, metadata_columns=["id"], page_content_columns=["content","media_type","test_metadata"]
    )
    
    DATASET = "my_langchain_dataset"  # @param {type: "string"}
    TABLE = "doc_and_vectors"  # @param {type: "string"}


    PROJECT_ID = 'nine-quality-test'

    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )

    print('you are here 1')
    store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET,
        table_name=TABLE,
        location=REGION,
        embedding=embedding,
    )
    
    print('you are here 2')

    documents = []
    documents.extend(loader.load())
    
    print('you are here 3')

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5#,
       # separators=["\n\n"],
    )
    doc_splits = text_splitter.split_documents(documents)

    print('you are here 4')


    # get current processing time to add it to metadata, datetime object containing current date and time
    now = datetime.now()

    # Add chunk number to metadata
    chunk_idx=0
    prev=doc_splits[0].metadata["id"]
    for idx, split in enumerate(doc_splits):
        split.metadata["process_time"]=now
        if prev==split.metadata["id"]:
           split.metadata["chunk"] = chunk_idx      
        else:
            chunk_idx=0
            split.metadata["chunk"] = chunk_idx
            prev=split.metadata["id"]
        chunk_idx +=1
        
    print('you are here 5')

    embedding_model = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )
    bq_store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET,
        table_name=TABLE,
        location=REGION,
        embedding=embedding,
    ) 
    print('you are here 6')


    _=bq_store.add_documents(doc_splits)

    return 'done'