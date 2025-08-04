from google.cloud import bigquery
from google.cloud.aiplatform_v1.types import NearestNeighborQuery
from vertexai.resources.preview import (FeatureOnlineStore, FeatureView,
                                        FeatureViewBigQuerySource)
from vertexai.resources.preview.feature_store import utils

#set project info
PROJECT_ID ='nine-quality-test'
REGION = "us-central1" 

#Verify that the FeatureView instance is created by getting the feature view.
FEATURE_ONLINE_STORE_ID = "nine_quality_test_multimodal_featurestore"  # @param {type: "string"}
FEATURE_VIEW_ID = "multimodal_feature_view_nine_quality_test"  # @param {type: "string"}
MM_nine_fv=FeatureView(
    FEATURE_VIEW_ID, feature_online_store_id=FEATURE_ONLINE_STORE_ID
) 
FEATURE_ONLINE_STORE_ID = "nine_quality_test_content_featurestore"  # @param {type: "string"}
FEATURE_VIEW_ID = "content_feature_view_nine_quality_test"  # @param {type: "string"}
content_nine_fv=FeatureView(
    FEATURE_VIEW_ID, feature_online_store_id=FEATURE_ONLINE_STORE_ID
) 

def get_response(feature_vector,converted_query_to_embedding,n_neighbor=10):
    result=feature_vector.search(
        embedding_value=converted_query_to_embedding,
        neighbor_count=n_neighbor,
        #string_filters=[country_filter],#for multimodal embedding this can be set to None, unless having a description column
        return_full_entity=True,  # returning entities with metadata
    )
    result=result.to_dict()
    return result

def get_neighbours(neighbours,embedding_type):
    nearest_neighbours=[]
    for neighbour in neighbours['neighbors']:
        nearest_neighbour={}
        nearest_neighbour['entity_id']=neighbour['entity_id']
        nearest_neighbour['distance']=neighbour['distance']

        for feature in neighbour['entity_key_values']['key_values']['features']:
            if 'value' in feature:
                if type(list(feature['value'].values())[0]) is dict:
                    nearest_neighbour[feature['name']]=[]#list(list(feature['value'].values())[0].values())[0]             
                else:
                    nearest_neighbour[feature['name']]=list(feature['value'].values())[0]             
            else :
                nearest_neighbour[feature['name']]=None
            nearest_neighbour['embedding_type']=embedding_type

        nearest_neighbours.append(nearest_neighbour)
    return nearest_neighbours

import streamlit as st
#from predict_request_gapic import *
import json
#import prediction client
from Embeddings.EmbeddingPredictionCLS import EmbeddingPredictionClient
neighbor_count=10

st.set_page_config(
     layout="wide",
     page_title="JS Lab",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"
)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "9News Search Engine POC"
)

st.title("9News Search Engine: Semantic Searches for Images and Videos")
st.subheader("Implemented by Intelia")


#client to access multimodal-embeddings model to convert text to embeddings
client = EmbeddingPredictionClient(project=PROJECT_ID)


scopes = ["https://www.googleapis.com/auth/cloud-platform"]
sa_file_path = "serious-hall-371508-b584a49b4817.json"
##credentials = service_account.Credentials.from_service_account_file(sa_file_path, scopes=scopes)
##client_options = { "api_endpoint": "1256643097.asia-southeast1-712487506603.vdb.vertexai.goog"}

# client to access GCS bucket
# #storage_client = storage.Client(credentials=credentials)
# #bucket = storage_client.bucket("js-multimodal-embeddings")

 

allResults=[]

search_term = 'I am searching for ' + st.text_input('Search: ')
if search_term !='I am searching for ':
    converted_query_to_mm_embedding = client.get_multimodal_embedding(text=search_term).text_embedding
    converted_query_to_text_embedding = client.get_text_embedding(text=search_term).text_embedding
    nearest_neighbours=[]
    
    result_mm=get_response(MM_nine_fv,converted_query_to_mm_embedding,neighbor_count)
    result_content=get_response(content_nine_fv,converted_query_to_text_embedding,neighbor_count)
   
    mm_nearest_neighbours=get_neighbours(result_mm,'multimodal')
    content_nearest_neighbours=get_neighbours(result_content,'content')
    allResults=mm_nearest_neighbours+content_nearest_neighbours

if len(allResults)>=1:
 st.write("")
 st.write("These are the most relevant results matching your search query:")
 st.image(allResults, width=200)
elif search_term =='I am searching for ':
 st.write("Please type in a search query above")
else:
 st.write("Sorry! There are no images matching your query. Please try again.")
