from google.cloud import aiplatform
import vertexai
import pandas as pd
import json
import math
from collections import Counter

from vertexai.evaluation import (
    EvalTask, 
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    CustomMetric
)

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

class PointWiseEvaluationClient:
    """Wrapper around Pointwise Evaluation Client."""

    def __init__(
        self,
        project: str=None,
        location: str = "us-central1",
        items: pd.core.frame.DataFrame = None,
        response_desc_column_name: str= 'description',
        response_llm_model_column_name: str= None,
        response_avgLogprobs_column_name: str=None,
        response_mediaType_column_name: str=None,
        response_media_column_metadata : dict=None,
        response_userPrompt_column_name: str=None,
        multimodal_evaluation_promt: dict=None,
        eval_metrics: list[dict] =None,
        experiment_name: str="pointwise-evaluation-experiment",
        evaluation_prompt: str="Evaluate the AI's contribution to a meaningful content generation",  
        delete_experiment: bool= True,
        sys_metrics: bool= True,
        ):
        """
        Initis the hyper parameters
        
        Args:
         str project:  project id 
         str locations: project location         
         Dataframe items: dataframe of AI-generated responses
         str response_desc_column_name: the name of the column in the 'items' dataframe that includes the AI-generated response
         str response_llm_model_column_name: the name of the column in the 'items' dataframe that includes the name of the model that is used for extracting AI-generated responses
         str response_avgLogprobs_column_name:  the name of the column in the 'items' dataframe that includes AI-generated response average probability log values
         str response_mediaType_column_name:  the name of the column in the 'items' dataframe that represent media type
         str response_userPrompt_column_name: the name of the column in the 'items' dataframe that represent user prompt using which the AI model generated the response
         dict response_media_column_metadata: dictionary including the name of fileuri, start and endoffset of the media if available
                                              e.g. {'fileUri':'fileUri', 'startOffset':'startOffset_seconds','endOffset':'endOffset_seconds', 'mediaType':'mediaType'}           
         dict multimodal_evaluation_promt: dictionary including prompts for multimodal content evaluations.
                                           e.g. {"video_prompt":"...","image_prompt":"..."}
         list[dict] eval_metrics: user defined evaluation metrics along with their rating rubric
                                  e.g.  [ {  "metric": "safety", "criteria": "..." }]
         str experiment_name: name of the evaluation experiment
         str evaluation_prompt: the prompt text which will be used as a prompt to evaluate the eval_metrics        
         bool delete_experiment: delete the generated experience after the evaluation are done if True. Will save costs.
         bool sys_metrics: calculates some mathematical metrics including perplexity, entropy if set to True.
        """
        
        #set the parameters
        self.location = location  
        self.project = project   
        self.items =items  
        self.eval_metrics=eval_metrics #user defined metrics along with their rubric ratings
        self.experiment_name=experiment_name
        self.evaluation_prompt=evaluation_prompt
        self.multimodal_evaluation_promt=multimodal_evaluation_promt
        self.response_userPrompt_column_name=response_userPrompt_column_name
        self.response_llm_model_column_name=response_llm_model_column_name
        self.response_media_column_metadata=response_media_column_metadata
        self.response_mediaType_column_name=response_mediaType_column_name
        self.response_desc_column_name=response_desc_column_name
        self.delete_experiment=delete_experiment
        self.response_avgLogprobs_column_name= response_avgLogprobs_column_name
        self.sys_metrics=sys_metrics
        self.run_experiment_name=self.experiment_name+"-"+ str(uuid.uuid4())
        
        #initialize Vertex AI
        vertexai.init(project=self.project, location= self.location )
         

    def set_evaluation_data(self):
        """
        Prepare the input data as in a dataframe for evaluation

        """
            
        eval_dataset = pd.DataFrame(
                        {
                            "response": self.items[self.response_desc_column_name].to_list(),
                            "evaluation_prompt":[self.evaluation_prompt]* len(self.items),                        
                            **({"mediaType": self.items[self.response_mediaType_column_name].to_list()} if 
                               self.response_mediaType_column_name !=None else {}),   
                            **({"avgLogprobs": self.items[self.response_avgLogprobs_column_name].to_list()} if 
                               self.response_avgLogprobs_column_name !=None else {}),
                            **({"multimodal_evaluation_promt": [
                                self.multimodal_evaluation_promt['video_prompt'] if 'video' in str(self.items[self.response_mediaType_column_name].to_list()[i]).lower() else 
                                self.multimodal_evaluation_promt['image_prompt'] if 'image' in str(self.items[self.response_mediaType_column_name].to_list()[i]).lower() else None
                                for i in range(len(self.items))
                            ]} if self.response_mediaType_column_name!=None and self.multimodal_evaluation_promt!=None else {}),
                       
                             **({"instruction": self.items[self.response_userPrompt_column_name].to_list()} if 
                               self.response_userPrompt_column_name !=None else {}),                            
                            
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
                            "response_llm_model": self.items[self.response_llm_model_column_name],
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

        table_id = config['pointwise_eval_table']
        dataset_id = config['eval_dataset']
        project_id = config["project"]
        location_id = config["project_location"]
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        dataset_full_id = f"{project_id}.{dataset_id}"

        # Remove unwanted characters from column names
        result.columns = result.columns.str.replace("/", "_")

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

    def perplexity(self,prob: float):    
        """Extract perplexity- models confidence in predicting next token using average log probablity

          Args:
          float prob: average log probability

          Returns:
          float:  perplexity value

          """
        return math.exp(-prob)
    
    
    def entropy(self,text: str):
        """Extracts entropy of a texts, higher entropy means diverse range of tokens have been choosen

        Args:
        str text: the input text

        Returns:
        float entropy: entropy value of input text
        """

        # Tokenize the text into words (ignoring punctuation)
        words = text.lower().split()

        # Get the frequency of each word
        word_count = Counter(words)

        # Total number of words
        total_words = len(words)

        # Calculate the probability of each word
        probabilities = [count / total_words for count in word_count.values()]

        # Calculate entropy using the formula
        entrpy = -sum(p * math.log2(p) for p in probabilities)

        return entrpy
    
    
    def get_autorater_response(self, metric_prompt: list, llm_model: str="gemini-1.5-pro") -> dict:
        
        """Extract evaluation metric on a AI-generated content using a AI-as-judge approach
        
        Args:
        list metric_prompt: the input metric prompt parameters
        str llm_model: evaluation model

        Returns:
        dict response_json: the evaluated metric in json format
        """
            
        # set evaluation metric schema
        metric_response_schema = {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "NUMBER"},
                "explanation": {"type": "STRING"},
            },
            "required": ["score", "explanation"],
        }

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
        dict : coverage evaluation
        """
        
        fileUri = json.loads(instance["reference"])["fileuri"]
        eval_instruction_template =instance["multimodal_evaluation_promt"] 
        response = instance["response"]
        user_prompt_instruction= instance["response"]
        
        evaluation_prompt=[]
        # set the evaluation prompt
        if 'video' in instance["mediaType"]:   
            evaluation_prompt = [
                eval_instruction_template,       
                "VIDEO URI: ",
                fileUri,
                "VIDEO METADATA: ",
                json.dumps(json.loads(instance["reference"])["metadata"]),  
                "USER'S INPUT PROMPT:",
                user_prompt_instruction,
                "GENERATED RESPONSE: ",
                response,
            ]
        elif 'image' in instance["mediaType"]:
            # generate the evaluation prompt
            evaluation_prompt = [
                eval_instruction_template,       
                "IMAGE URI: ",
                fileUri,   
                "USER'S INPUT PROMPT:",
                user_prompt_instruction,
                "GENERATED RESPONSE: ",
                response,
            ]
     
        #generate evaluation response
        evaluation_response = self.get_autorater_response(evaluation_prompt)
        return {
           "custom_coverage":  evaluation_response.get("score", ""),
             "explanation": evaluation_response.get("explanation", "") 
        }
    
    def get_evaluations(self):
        """
        Extracts the evaluation metricsusing:
            1-user defined metrics and rating criteria
            2-pre-defined mathematical metrics: perplexity, entropy

        """
        metrics=[]
        # set evaluation data
        eval_dataset=self.set_evaluation_data()
        
        #calculate the system defined metrics
        if self.sys_metrics:
            # the evrage prob column is given in the data, calculate perplexity
            if self.response_avgLogprobs_column_name:
                eval_dataset['perplexity']=eval_dataset[self.response_avgLogprobs_column_name].apply(self.perplexity)
            
            #calculate entropy
            eval_dataset['entropy']=eval_dataset['response'].apply(self.entropy)
            eval_results=eval_dataset
            
        #calculate coverage metrics
        if self.multimodal_evaluation_promt:
            
            #create a custome coverage metric
            custom_coverage_metric = CustomMetric(
                name="custom_coverage",
                metric_function=self.custom_coverage_fn,
            )
            
            metrics. append(custom_coverage_metric)
 


        #set user defined metrics
        if self.eval_metrics:   
            # Define  pointwise quality metric(s)
            for metric in self.eval_metrics:
                # Define a pointwise quality metric
                pointwise_quality_metric_prompt = f"""{self.evaluation_prompt}; evaluate {metric['metric']}.
                # Rubric rating criteria
                {metric['criteria']}
                # AI-generated Response
                {{response}}
                """
                pointwise_metric=PointwiseMetric(
                    metric=metric['metric'],
                    metric_prompt_template=pointwise_quality_metric_prompt,
                )
                metrics.append(pointwise_metric)
                
        #if any metric is defined define task and extract metrics
        if metrics:
            # Create the evaluation task
            eval_task = EvalTask(
                dataset=eval_dataset,
                metrics=metrics,
                experiment=self.experiment_name,
            )
            # Run evaluation on the data using the evaluation service
            results = eval_task.evaluate( 

                    experiment_run_name=self.run_experiment_name,
                ) 
            #Delete the experiment after getting the result
            if self.delete_experiment:
                experiment = aiplatform.Experiment(self.experiment_name)
                experiment.delete()
                
            eval_results=results.metrics_table
            
        #log the statistics into bigquery
        self.log_evaluations(eval_results)
            
        return eval_results 

    