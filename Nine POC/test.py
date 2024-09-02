from google.cloud import bigquery
from google.cloud.aiplatform_v1.types import NearestNeighborQuery
from vertexai.resources.preview import (FeatureOnlineStore, FeatureView,
                                        FeatureViewBigQuerySource)
from vertexai.resources.preview.feature_store import utils

import streamlit as st
#from predict_request_gapic import *
import json
#import prediction client
from Embeddings.EmbeddingPredictionCLS import EmbeddingPredictionClient
neighbor_count=10
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import requests

#set project info
PROJECT_ID ='nine-quality-test'
REGION = "us-central1" 
 
try:
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

except:
    print('create online store...It does not exists or has been deleted.')
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
   
def load_image(image_path):
    """Load image from a URL or file path."""
    try:
        if image_path.startswith('SampleImage/'):
            image = Image.open(image_path)
        else:
            # Handling Google Cloud Storage URLs or other remote paths
            import requests
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None
    


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

#client to access GCS bucket
##storage_client = storage.Client(credentials=credentials)
##bucket = storage_client.bucket("js-multimodal-embeddings")

 

allResults=[]

search_term = 'I am searching for ' + st.text_input('Search: ')
if search_term !='I am searching for ':
    allResults=[{'entity_id': '60MI24_1_A_HBB.mp4|2400|2416',
  'distance': -0.1292482316493988,
  'end_offset_sec_embedding': '2416',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2400',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2416|2432',
  'distance': -0.12086805701255798,
  'end_offset_sec_embedding': '2432',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2416',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '60MI24_1_A_HBB.mp4|1936|1952',
  'distance': -0.1192726120352745,
  'end_offset_sec_embedding': '1952',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '1936',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2392|2400',
  'distance': -0.11390428990125656,
  'end_offset_sec_embedding': '2400',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2392',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2088|2104',
  'distance': -0.11389684677124023,
  'end_offset_sec_embedding': '2104',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2088',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2160|2176',
  'distance': -0.11209405958652496,
  'end_offset_sec_embedding': '2176',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2160',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': '7944pcnf.png',
  'distance': -0.11018449068069458,
  'end_offset_sec_embedding': None,
  'embedding_type': 'multimodal',
  'media_type': 'image',
  'start_offset_sec_embedding': None,
  'multimodal_embedding': [],
  'path': 'SampleImage/7944pcnf.png'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2104|2120',
  'distance': -0.10899774730205536,
  'end_offset_sec_embedding': '2120',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2104',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': 'pnll2ztl.png',
  'distance': -0.1062314510345459,
  'end_offset_sec_embedding': None,
  'embedding_type': 'multimodal',
  'media_type': 'image',
  'start_offset_sec_embedding': None,
  'multimodal_embedding': [],
  'path': 'SampleImage/pnll2ztl.png'},
 {'entity_id': '60MI24_1_A_HBB.mp4|2376|2392',
  'distance': -0.10622908174991608,
  'end_offset_sec_embedding': '2392',
  'embedding_type': 'multimodal',
  'media_type': 'video',
  'start_offset_sec_embedding': '2376',
  'multimodal_embedding': [],
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4'},
 {'entity_id': 'pnll2ztl.png',
  'distance': -0.6711065769195557,
  'end_offset_sec_chapter': None,
  'embedding_type': 'content',
  'start_offset_sec_chapter': None,
  'media_type': 'image',
  'content': " Donald Trump is giving a speech with a microphone in his right hand. He is wearing a white shirt and dark suit jacket. He has a serious expression on his face. There is a large American flag in the background. There are two men and one woman behind him. The woman is wearing a black dress and has her hand on Trump's shoulder. One man is wearing a black suit and sunglasses. The other man is wearing a white shirt and black suit jacket.",
  'path': 'SampleImage/pnll2ztl.png',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2960|2976',
  'distance': -0.6441819667816162,
  'end_offset_sec_chapter': '2976',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2960',
  'media_type': 'video',
  'content': 'Joe Biden, current president of the US, is also facing age concerns, as he is the oldest US president in history.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '7944pcnf.png',
  'distance': -0.6375225782394409,
  'end_offset_sec_chapter': None,
  'embedding_type': 'content',
  'start_offset_sec_chapter': None,
  'media_type': 'image',
  'content': " The image shows Joe Biden, a man with white hair and a blue suit, standing next to a young girl with blonde hair and a white dress. The girl is smiling and has her arm around Biden's waist. In the background is the White House. On the right is a photo of Hunter Biden, Joe Biden's son.",
  'path': 'SampleImage/7944pcnf.png',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2976|2992',
  'distance': -0.6308561563491821,
  'end_offset_sec_chapter': '2992',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2976',
  'media_type': 'video',
  'content': 'It is interesting to see that even though both Trump and Biden are very old, they are both running for president again. Biden is the oldest US president in history, while Trump will be 78 years old when the 2024 election rolls around.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2792|2808',
  'distance': -0.6274047493934631,
  'end_offset_sec_chapter': '2808',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2792',
  'media_type': 'video',
  'content': 'The man looks directly at the camera.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '816f62dz.png',
  'distance': -0.6194294691085815,
  'end_offset_sec_chapter': None,
  'embedding_type': 'content',
  'start_offset_sec_chapter': None,
  'media_type': 'image',
  'content': ' The image is a portrait of a young male with short brown hair and glasses. He is wearing a gray shirt with an American flag design on the front. He has a slight smile on his face and is looking at the camera. The background is a dark blue.',
  'path': 'SampleImage/816f62dz.png',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2880|2896',
  'distance': -0.6191455125808716,
  'end_offset_sec_chapter': '2896',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2880',
  'media_type': 'video',
  'content': 'Donald Trump, former president of the US, is facing 91 indictments and 4 criminal trials. The charges relate to his attempts to overturn the 2020 election results.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2992|3000',
  'distance': -0.6039376258850098,
  'end_offset_sec_chapter': '3000',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2992',
  'media_type': 'video',
  'content': 'People are already concerned about age being a factor for both Trump and Biden, as they both have shown signs of forgetfulness while in the White House.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2928|2944',
  'distance': -0.5939319133758545,
  'end_offset_sec_chapter': '2944',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2928',
  'media_type': 'video',
  'content': "Trump's supporters believe that he is a rockstar superhero president, a hero to democracy. They believe that the charges against him are a political witch hunt by Joe Biden and the Democrats.",
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []},
 {'entity_id': '60MI24_1_A_HBB.mp4|2944|2960',
  'distance': -0.5932260751724243,
  'end_offset_sec_chapter': '2960',
  'embedding_type': 'content',
  'start_offset_sec_chapter': '2944',
  'media_type': 'video',
  'content': 'Trump is facing challenges in the Republican Party, and he is now 78 years old, which could be a factor in the 2024 election.',
  'path': 'gs://raw_nine_files/60MI24_1_A_HBB.mp4',
  'content_embedding': []}]
        
if len(allResults)>=1:
    results=allResults
    st.write("")
    st.write("These are the most relevant results matching your search query:")
   
    df = pd.DataFrame(results)

    # Display results
    st.subheader('Results:')
    # Display in a grid format
    num_columns = 3
    columns = st.columns(num_columns)
    
    for idx, result in enumerate(results):
        col_idx = idx % num_columns
        with columns[col_idx]:
            st.write(f"**Entity ID:** {result.get('entity_id')}")
            st.write(f"**Distance:** {result.get('distance')}")
            
            media_type = result.get('media_type')
            media_path = result.get('path')

            if media_type == 'image':
                image = load_media(media_path, media_type)
                if image:
                    st.image(image, caption=result.get('entity_id'))
            elif media_type == 'video':
                st.video(media_path)  # Display video using Streamlit's built-in video player
            
            st.write("-----")
            
elif search_term =='I am searching for ':
 st.write("Please type in a search query above")
else:
 st.write("Sorry! There are no images matching your query. Please try again.")