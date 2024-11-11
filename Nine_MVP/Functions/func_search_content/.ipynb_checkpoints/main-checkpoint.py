
import functions_framework
import time
import random
from EmbeddingPredictionClient import EmbeddingPredictionClient  
from google.cloud import bigquery
import json
import asyncio

async def exponential_backoff_retries(client, text=None, image_file=None, max_retries=5, embedding_type=None):
    """
    This function applies exponential backoff with retries to the API calls.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Try to get the embedding from the client
            if embedding_type=="multimodal_embedding":
                    return client.get_multimodal_embedding(text, image_file)
            elif embedding_type=="text_embedding":
                    return client.get_text_embedding(text)
        except Exception as e:
            attempt += 1
            backoff_delay = min(2 ** attempt + random.uniform(0, 1), 32)  # Exponential backoff with jitter
            print(f"Attempt {attempt} failed with error {e}. Retrying in {backoff_delay:.2f} seconds...")
            time.sleep(backoff_delay)  # Wait before retrying

    raise Exception("Max retries reached. Could not complete the request.")

    
async def generate_query_embedding(client,text=None,image_file=None, embedding_type=None):
    try:
        # Retry logic with exponential backoff to calculate query embeddings
        result = exponential_backoff_retries(embedding_client, text, image_file, embedding_type)
        
        # Respond with the successful embedding response
        return {
            "text_embedding": result.text_embedding,
            "image_embedding": result.image_embedding
        }, 200

    except Exception as e:
        # Handle failure after max retries
        return f"Error: {str(e)}", 500


async def get_media_nearest_neighbors(query_embedding, table, dataset,source_embedding_column,project_id,top_k=50):
    """Query nearest neighbors using cosine similarity in BigQuery for multimodal embeddings."""
    
    # Record the start time
    start_time = time.time()
    #option="""'{"fraction_lists_to_search": 0.01}'"""
    sql = f"""  
         WITH search_results AS
         (
              SELECT
              search_results.base.uri as uri,  
              search_results.base.combined_multimodal_id as combined_id,
              search_results.distance,  -- The computed distance (similarity score) between the embeddings
              search_results.base.asset_id ,
              ROW_NUMBER() OVER (PARTITION BY search_results.base.asset_id ORDER BY distance) AS rank_within_document  -- Rank by distance within each document
              
            FROM
              VECTOR_SEARCH(     
                TABLE `{dataset}.{table}`, --source embedding table
                '{source_embedding_column}',  -- Column with the embedding vectors in the base table

                -- Use the query embedding computed in the previous step
                 (SELECT {json.dumps(query_embedding)} query_embedding),  -- The query embedding from the CTE (query_embedding)

                -- Return top-k closest matches (adjust k as necessary)
                top_k =>{ top_k  }, -- Top k most similar matches based on distance
                distance_type => 'COSINE'                 
              ) search_results
              
          )
          -- Step 2: Aggregate relevance per document (original_document_id)
            ,aggregated_results AS (
                SELECT
                    asset_id,
                    COUNT(*) AS chunk_count,  -- The number of chunks for this document
                    SUM(distance) AS total_distance,  -- Sum of the distances for this document's chunks
                    AVG(distance) AS avg_distance  -- Alternatively, you can use the average distance
                FROM search_results
                GROUP BY asset_id
            ),

            -- Step 3: Rank the documents by relevance (number of chunks and sum of distances)
            ranked_documents AS (
                SELECT
                    asset_id,
                    chunk_count,
                    total_distance,
                    avg_distance,
                    ROW_NUMBER() OVER (ORDER BY chunk_count DESC, total_distance ASC) AS final_rank  -- Rank by chunk_count and then distance

                FROM aggregated_results
            )

            -- Step 4: Retrieve the top-k ranked documents based on relevance
            SELECT * FROM (
              SELECT  
                sr.asset_id,  
                sr.uri,              
                ROW_NUMBER() OVER (PARTITION BY SR.asset_id) AS IDX,
               -- sr.distance,
                final_rank--,
               -- rank_within_document
            FROM search_results sr
            JOIN ranked_documents rd ON sr.asset_id = rd.asset_id
            WHERE rd.final_rank <= {top_k} -- Return the top-k documents based on chunk relevance
            ORDER BY rd.final_rank, sr.rank_within_document  -- Order by document relevance and chunk rank
            )
            WHERE IDX=1
    """       
    #print(sql)
    bq_client = bigquery.Client(project_id)
  
    # Run the query
    query_job = bq_client.query(sql)

    # Fetch results
    results = query_job.result()
    
    output=[]
    for row in results:
        output.append({'asset_id':row['asset_id'], 'uri':row['uri']})

    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(elapsed_time)
    return output

async def get_content_nearest_neighbors(query_embedding, table, dataset,source_embedding_column,project_id,top_k=50):
    """Query nearest neighbors using cosine similarity in BigQuery for text embeddings."""
    
    # Record the start time
    start_time = time.time()
    #option="""'{"fraction_lists_to_search": 0.01}'"""
    sql = f"""  
         WITH search_results AS
         (
              SELECT
              search_results.base.content as content,  
              search_results.base.combined_id as combined_id,
              search_results.distance,  -- The computed distance (similarity score) between the embeddings
              b.asset_id,
              b.headline,
              b.description,
              ROW_NUMBER() OVER (PARTITION BY b.asset_id ORDER BY distance) AS rank_within_document  -- Rank by distance within each document
              
            FROM
              VECTOR_SEARCH(     
                TABLE `{dataset}.{table}`, --source embedding table
                '{source_embedding_column}',  -- Column with the embedding vectors in the base table

                -- Use the query embedding computed in the previous step
                 (SELECT {json.dumps(query_embedding)} query_embedding),  -- The query embedding from the CTE (query_embedding)

                -- Return top-k closest matches (adjust k as necessary)
                top_k =>{ top_k  }, -- Top k most similar matches based on distance
                distance_type => 'COSINE'                  
              ) search_results
              --this part should be removed later
              inner join   `nine-quality-test.vlt_media_content_prelanding.vlt_combined_media_content` b
              on search_results.base.combined_id =b.combined_id  
          )
          -- Step 2: Aggregate relevance per document (original_document_id)
            ,aggregated_results AS (
                SELECT
                    asset_id,
                    COUNT(*) AS chunk_count,  -- The number of chunks for this document
                    SUM(distance) AS total_distance,  -- Sum of the distances for this document's chunks
                    AVG(distance) AS avg_distance  -- Alternatively, you can use the average distance
                FROM search_results
                GROUP BY asset_id
            ),

            -- Step 3: Rank the documents by relevance (number of chunks and sum of distances)
            ranked_documents AS (
                SELECT
                    asset_id,
                    chunk_count,
                    total_distance,
                    avg_distance,
                    ROW_NUMBER() OVER (ORDER BY chunk_count DESC, total_distance ASC) AS final_rank  -- Rank by chunk_count and then distance

                FROM aggregated_results
            )

            -- Step 4: Retrieve the top-k ranked documents based on relevance
            SELECT * FROM (
              SELECT  
                sr.asset_id,  
                sr.headline,
                sr.description,
                sr.combined_id,
                ROW_NUMBER() OVER (PARTITION BY SR.asset_id) AS IDX,
                sr.distance,
                final_rank--,
               -- rank_within_document
            FROM search_results sr
            JOIN ranked_documents rd ON sr.asset_id = rd.asset_id
            WHERE rd.final_rank <= {top_k} -- Return the top-k documents based on chunk relevance      
            --and sr.asset_id like '%00261507986b0faf31c775597d2d24beb4381e43%'
            ORDER BY rd.final_rank, sr.rank_within_document  -- Order by document relevance and chunk rank
            )
            WHERE IDX=1
    """       
    #print(sql)
    bq_client = bigquery.Client(project_id)
  
    # Run the query
    query_job = bq_client.query(sql)

    # Fetch results
    results = query_job.result()
    
    output=[]
    for row in results:
        output.append({'asset_id':row['asset_id'], 'headline':row['headline'],'description':row['description']})

    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(elapsed_time)
    return output

def merge_result(combined_list):
    # Step 2: Create a dictionary to merge by 'id'
    merged_dict = {}

    # Step 3: Iterate through the combined list and merge dictionaries by 'id'
    for d in combined_list:
        id_value = d['asset_id']

        # If the id already exists in merged_dict, update it
        if id_value in merged_dict:
            merged_dict[id_value].update(d)
        else:
            # If the id doesn't exist, add the dictionary as it is
            merged_dict[id_value] = d.copy()

    # Step 4: Convert the merged dictionary back into a list
    final_merged_list = list(merged_dict.values())
    
    return final_merged_list



async def get_nearest_contet(request):
    """
    Cloud Function entry point. This function handles the incoming request, 
    performs exponential backoff retries, and returns the embedding response.
    """
    # Parse the incoming request to extract text or image file
    # request_json = request.get_json(silent=True)
    # text = request_json.get('text')
    # image_file = request_json.get('image_file')  # Assume it's the path or base64 string of the image
    # project = request_json.get('project')  
    # region = request_json.get('region')  

    project_id='nine-quality-test'
    region="us-central1"
    text='Curtis Sittenfeld'
    image_file=None
    
    top_k=50     
    dataset='langchain_dataset'
    content_table='vlt_media_content_text_test_for_search'
    mm_table='vlt_imgvdo_multimodal_embeddings'
    source_embedding_column='ml_generate_embedding_result'

    # Initialize the EmbeddingPredictionClient outside the function for reuse
    embedding_client = EmbeddingPredictionClient(project=project_id , location=region,api_regional_endpoint=region+"-aiplatform.googleapis.com")
        
    if not text and not image_file:
        print('you are here')
        return 'Error: At least one of "text" or "image_file" must be provided.', 400
     
    content_result=[]
    media_text_result=[]
    media_image_result=[]
    if text:
        #if a text is given, calculate both multiomdal embedding and text embedding of the search query
        txtembding_for_text_result =  await asyncio.create_task(exponential_backoff_retries(embedding_client, text, embedding_type='text_embedding'))
        mmembding_for_text_result =  await asyncio.create_task(exponential_backoff_retries(embedding_client, text, embedding_type='multimodal_embedding')) 
        txtembding_for_text_result=txtembding_for_text_result .text_embedding
        mmembding_for_text_result=mmembding_for_text_result.text_embedding
        #find nearest neighbours
        content_result = await asyncio.create_task(get_content_nearest_neighbors(txtembding_for_text_result, content_table, dataset,source_embedding_column,project_id,top_k=top_k))
        dataset='vlt_media_embeddings_integration'
        media_text_result = await asyncio.create_task(get_media_nearest_neighbors(mmembding_for_text_result, mm_table, dataset,source_embedding_column,project_id,top_k=top_k))
        print('search is done')
            
    if image_file:
        #if an image is given convert image to 64bytestring and extract embedding
        mmembding_for_image_result = await asyncio.create_task(exponential_backoff_retries(embedding_client, image_file, embedding_type='multimodal_embedding').image_embedding)
        dataset='vlt_media_embeddings_integration'
        media_image_result = await asyncio.create_task(get_media_nearest_neighbors(mmembding_for_image_result, mm_table, dataset,source_embedding_column,project_id,top_k=top_k))
        media_image_result=media_image_result
        
    final_merged_list=merge_result(content_result+media_text_result+media_image_result)
    return final_merged_list#, content_result, media_text_result, media_image_result

# @functions_framework.http
# async def search_content_function(request):
 
#     result = await get_nearest_contet(request) 
#     return result#[0],result[1],result[2]

@functions_framework.http
def search_content_function(request):
    """This is the entry point for the Cloud Function."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        # If no event loop is running, create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result = loop.run_until_complete(get_nearest_contet(request))
    return result
     