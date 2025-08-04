from google.cloud import aiplatform
import vertexai
import pandas as pd
import json
import math
from collections import Counter
import uuid 
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from datetime import datetime

from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part as GenerativeModelPart,
    HarmBlockThreshold,
    HarmCategory,
)

class PairwiseEvaluationClient:
    """Wrapper around Pairwise Evaluation Client."""

    def __init__(
        self,
        project: str=None,
        location: str = "us-central1",
        items: pd.core.frame.DataFrame = None,
        response_A_desc_column_name: str= 'description_A',
        response_B_desc_column_name: str= 'description_B',
        response_A_llm_model_column_name: str= None,
        response_B_llm_model_column_name: str=None,
        response_mediaType_column_name: str=None,
        response_media_column_metadata : dict=None,
        response_A_userPrompt_column_name: str=None,
        response_B_userPrompt_column_name: str=None,
        multimodal_evaluation_promt: dict=None,       
        experiment_name: str="pairwise-evaluation-experiment",
       
        ):
        """
        Initis the hyper parameters
        
        Args:
         str project:  project id 
         str locations: project location         
         Dataframe items: dataframe of AI-generated responses
         str response_A_desc_column_name: the name of the column in the 'items' dataframe that includes the AI-generated response for model A
         str response_B_desc_column_name: the name of the column in the 'items' dataframe that includes the AI-generated response for model B
         str response_A_llm_model_column_name: the name of the column in the 'items' dataframe that includes the model A's name that is used for extracting AI-generated responses A
         str response_B_llm_model_column_name: the name of the column in the 'items' dataframe that includes the model B's name that is used for extracting AI-generated responses B
         str response_mediaType_column_name:  the name of the column in the 'items' dataframe that represent media type
         str response_A_userPrompt_column_name: the name of the column in the 'items' dataframe that represent user prompt for model A using which the AI model generated the response A
         str response_A_userPrompt_column_name: the name of the column in the 'items' dataframe that represent user prompt for model B using which the AI model generated the response B
         dict response_media_column_metadata: dictionary including the name of fileuri, start and endoffset of the media if available
                                              e.g. {'fileUri':'fileUri', 'startOffset':'startOffset_seconds','endOffset':'endOffset_seconds', 'mediaType':'mediaType'}           
         dict multimodal_evaluation_promt: dictionary including prompts for multimodal content evaluations.
                                           e.g. {"video_prompt":"...","image_prompt":"..."}        
         str experiment_name: name of the evaluation experiment
        """
        
        #set the parameters
        self.location = location  
        self.project = project   
        self.items =items  
        self.experiment_name=experiment_name      
        self.multimodal_evaluation_promt=multimodal_evaluation_promt
        self.response_A_userPrompt_column_name=response_A_userPrompt_column_name
        self.response_B_userPrompt_column_name=response_B_userPrompt_column_name
        self.response_A_llm_model_column_name=response_A_llm_model_column_name
        self.response_B_llm_model_column_name=response_B_llm_model_column_name        
        self.response_media_column_metadata=response_media_column_metadata
        self.response_mediaType_column_name=response_mediaType_column_name
        self.response_A_desc_column_name=response_A_desc_column_name
        self.response_B_desc_column_name=response_B_desc_column_name
      
        self.run_experiment_name=self.experiment_name+"-"+ str(uuid.uuid4())

         # Load the schema from PairWise_Schema.json
        with open('PairWise_Schema.json') as config_file:
            self.pairwise_schema = json.load(config_file)
        
        #initialize Vertex AI
        vertexai.init(project=self.project, location= self.location )
         

    def set_evaluation_data(self):
        """
        Prepare the input data as in a dataframe for evaluation

        """
            
        eval_dataset = pd.DataFrame(
                        {
                            "response_A": self.items[self.response_A_desc_column_name].to_list(),
                            "response_B": self.items[self.response_B_desc_column_name].to_list(),
                                     
                            **({"mediaType": self.items[self.response_mediaType_column_name].to_list()} if 
                               self.response_mediaType_column_name !=None else {}),
                            **({"multimodal_evaluation_promt": [
                                self.multimodal_evaluation_promt['video_prompt'] if 'video' in str(self.items[self.response_mediaType_column_name].to_list()[i]).lower() else 
                                self.multimodal_evaluation_promt['image_prompt'] if 'image' in str(self.items[self.response_mediaType_column_name].to_list()[i]).lower() else None
                                for i in range(len(self.items))
                            ]} if self.response_mediaType_column_name!=None and self.multimodal_evaluation_promt!=None else {}),
                       
                             **({"instruction_A": self.items[self.response_A_userPrompt_column_name].to_list()} if 
                               self.response_A_userPrompt_column_name !=None else {}),   
                            **({"instruction_B": self.items[self.response_B_userPrompt_column_name].to_list()} if 
                               self.response_B_userPrompt_column_name !=None else {}),  
                            
                            "reference": [
                                        json.dumps(
                                            {
                                                "fileuri": self.items[self.response_media_column_metadata['fileUri']].to_list()[i],
                                                "metadata": {
                                                    "start_offset": {
                                                        "seconds": int(self.items[self.response_media_column_metadata['startOffset']].to_list()[i]),
                                                        "nanos": 0,
                                                    },
                                                    "end_offset": {
                                                        "seconds": int(self.items[self.response_media_column_metadata['endOffset']].to_list()[i]),
                                                        "nanos": 0,
                                                    },
                                                } if self.response_media_column_metadata['startOffset'] in self.items.columns and 
                                                     self.response_media_column_metadata['endOffset'] in self.items.columns and 
                                                     'video' in str(self.items[self.response_mediaType_column_name].to_list()[i]).lower() 
                                                else {}
                                            }
                                        ) if self.response_media_column_metadata is not None and 
                                             self.response_media_column_metadata.get('fileUri') is not None 
                                          else "{}"
                                
                                        for i in range(len(self.items))
                                    ],
                            "response_A_llm_model": self.items[self.response_A_llm_model_column_name],
                            "response_B_llm_model": self.items[self.response_B_llm_model_column_name],
                            "run_experiment_name": [self.run_experiment_name] * len(self.items),
                            "run_experiment_date": [datetime.today().strftime('%Y-%m-%d')] * len(self.items),
                        }
                    )
        
        return eval_dataset
    
    def log_evaluations(self,result):
        """
        Log the evaluation result into BigQuery, converting all columns to string type.

        Args:
            dataframe result : The evaluation result to be recorded into the database.
        """
        import json
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound

        # Load configuration from config.json
        with open('config.json') as config_file:
            config = json.load(config_file)

        table_id = config['pairwise_eval_table']
        dataset_id = config['eval_dataset']
        project_id = config["project"]
        location_id = config["project_location"]
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        dataset_full_id = f"{project_id}.{dataset_id}"

        # Remove unwanted characters from column names
        result.columns = result.columns.str.replace("/", "_").str.replace(',','')

        # Convert all columns to string
        result = result.astype(str)

        # Convert DataFrame to list of dictionaries
        data_as_dict = result.to_dict(orient='records')

        # Initialize BigQuery Client
        client = bigquery.Client()


        try:
            client.get_dataset(dataset_full_id)
            print(f"Dataset {dataset_full_id} exists.")
        except NotFound:
            print(f"Dataset {dataset_full_id} not found. Creating dataset...")
            dataset = bigquery.Dataset(dataset_full_id)
            dataset.location = location_id
            client.create_dataset(dataset)
            print(f"Dataset {dataset_full_id} created successfully.")


        # Ensure the dataset exists    
        try:
            # Fetch the existing table
            table = client.get_table(table_full_id)
            existing_schema = {field.name: field.field_type for field in table.schema}
            print(f"Table {table_full_id} exists. Checking schema...")

            # Identify new columns to be added
            schema_changes = []
            for col in result.columns:
                if col not in existing_schema:
                    schema_changes.append(bigquery.SchemaField(col, bigquery.enums.SqlTypeNames.STRING))

            if schema_changes:
                print("Altering schema to add new columns...")
                table.schema = table.schema + schema_changes
                table = client.update_table(table, ["schema"])
                print(f"Schema updated successfully.")

        except NotFound:
            print(f"Table {table_full_id} not found. Creating table...")
            # Define schema as all string types
            schema = [bigquery.SchemaField(name, bigquery.enums.SqlTypeNames.STRING) for name in result.columns]

            # Create the table
            table = bigquery.Table(table_full_id, schema=schema)
            table = client.create_table(table)
            print(f"Table {table_full_id} created successfully.")


        # Insert rows into BigQuery
        try:
            errors = client.insert_rows_json(table_full_id, data_as_dict)
            if not errors:
                print(f"Evaluations have successfully been loaded into {table_full_id}.")
            else:
                print("Errors occurred while loading data:")
                for error in errors:
                    print(error)
        except Exception as e:
            print(f"An error occurred while inserting data: {e}")
    
    
    def get_autorater_response(self, metric_prompt: list, llm_model: str="gemini-1.5-pro") -> dict:
        
        """Extract evaluation metric on a AI-generated content using a AI-as-judge approach
        
        Args:
        list metric_prompt: the input metric prompt parameters
        str llm_model: evaluation model

        Returns:
        dict response_json: the evaluated metric in json format
        """
            
        # set evaluation metric schema
        metric_response_schema = self.pairwise_schema 

        #define a generative model as an autorator
        autorater = GenerativeModel(
            llm_model,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=metric_response_schema,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        #generate the rating metrics as per requested measures and metric in the prompt
        response = autorater.generate_content(metric_prompt)

        response_json = {}

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if (
                candidate.content
                and candidate.content.parts
                and len(candidate.content.parts) > 0
            ):
                part = candidate.content.parts[0]
                if part.text:
                    response_json = json.loads(part.text)

        return response_json

    def custom_coverage_fn(self,instance):
       
        """Extract evaluation metric on a AI-generated content using a AI-as-judge approach
        
        Args:
        dict instance: an instance of predictions that should be evaluated
       
        Returns:
        dict evaluation_response: scores and explanations related to the judgements for each requested metric
        """
        
        fileUri = json.loads(instance["reference"])["fileuri"]
        eval_instruction_template =instance["multimodal_evaluation_promt"]      
        user_prompt_A_instruction= instance["instruction_A"]
        user_prompt_B_instruction= instance["instruction_B"]
        response_A = instance["response_A"]
        response_B = instance["response_B"]
        
        evaluation_prompt=[]
        # set the evaluation prompt
        if 'video' in instance["mediaType"]:   
            evaluation_prompt = [
                eval_instruction_template,       
                "VIDEO URI: ",
                fileUri,
                "VIDEO METADATA: ",
                json.dumps(json.loads(instance["reference"])["metadata"]),  
                "USER'S INPUT PROMPT MODEL A:",
                user_prompt_A_instruction,
                "USER'S INPUT PROMPT MODEL B:",
                user_prompt_B_instruction,
                "GENERATED RESPONSE MODEL A: ",
                 response_A,
                 "GENERATED RESPONSE MODEL B: ",
                 response_B,
            ]
        elif 'image' in instance["mediaType"]:
            # generate the evaluation prompt
            evaluation_prompt = [
                eval_instruction_template,       
                "IMAGE URI: ",
                fileUri,   
                "USER'S INPUT PROMPT MODEL A:",
                user_prompt_A_instruction,
                "USER'S INPUT PROMPT MODEL B:",
                user_prompt_B_instruction,
                "GENERATED RESPONSE MODEL A: ",
                 response_A,
                 "GENERATED RESPONSE MODEL B: ",
                 response_B,
            ]
     
        #generate evaluation response
        evaluation_response = self.get_autorater_response(evaluation_prompt)
        return evaluation_response

    # Function to extract the score and explanation for each category
    def flatten_evaluations(self,instance):
        """Flattens a dict column type in a dataframe series
        
        Args:
        pandas.core.series.Series instance: an instance of predictions that should be evaluated
       
        Returns:
        Dataframe flattened_data: flattened data
        """ 
        flattened_data = {}
        for key in self.pairwise_schema['required']:
            flattened_data[f"{key.lower().replace(' ', '_')}_score"] = instance[key]['score']
            flattened_data[f"{key.lower().replace(' ', '_')}_explanation"] = instance[key]['explanation']
        
        return flattened_data
  
    
    def get_evaluations(self):
        """
        Extracts the evaluation metricsusing:
            1-user defined metrics and rating criteria
            
        """
        # set evaluation data
        eval_dataset=self.set_evaluation_data()       
            
        #calculate coverage metrics
        if self.multimodal_evaluation_promt:
            #get evaluations
            eval_dataset['custom_coverage']=eval_dataset.apply(self.custom_coverage_fn,axis=1)
             
            # Apply the function to flatten the 'custom_coverage' column and create new columns
            flattened_df = eval_dataset['custom_coverage'].apply(self.flatten_evaluations)
                                                                 
            # Join the flattened columns to the original dataframe
            eval_dataset = eval_dataset.join(pd.json_normalize(flattened_df))
            eval_dataset = eval_dataset.drop(columns=["custom_coverage"])
            
        eval_results=eval_dataset
            
        #log the statistics into bigquery
        self.log_evaluations(eval_results)
            
        return eval_results 

    
