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
        self.response_llm_model_column_name=response_llm_model_column_name
        self.response_desc_column_name=response_desc_column_name
        self.delete_experiment=delete_experiment
        self.response_avgLogprobs_column_name= response_avgLogprobs_column_name
        self.sys_metrics=sys_metrics
        self.run_experiment_name=self.experiment_name+"-"+ str(uuid.uuid4())
        
        #initialize Vertex AI
        vertexai.init(project=self.project, location= self.location )
         

    def set_evaluation_data(self):
        """
        Sets the input data as in a dataframe for evaluation

        """
            
        eval_dataset= pd.DataFrame(
                                {
                                   # "instruction": instructions,
                                   # "context": contexts,
                                    "response": self.items[self.response_desc_column_name].to_list(),
                                    **({"avgLogprobs": self.items[self.response_avgLogprobs_column_name].to_list()} if 
                                                       self.response_avgLogprobs_column_name !=None else {}),
                                    "response_llm_model":[self.response_llm_model_column_name]*len(self.items),
                                    "run_experiment_name":[self.run_experiment_name]*len(self.items),
                                    "run_experiment_date" :  pd.to_datetime( [datetime.today().date()]*len(self.items)).\
                                                             strftime('%Y-%m-%d'),

                                }
                            )
         
        #eval_dataset['run_experiment_date'] = pd.to_datetime(eval_dataset['run_experiment_date']).dt.strftime('%Y-%m-%d')
        
        return eval_dataset

    def log_evaluations(self, result):
        """
        Log the evaluation result into BigQuery, altering the table schema if needed.

        Args:
            dataframe result : The evaluation result to be recorded into the database.
        """
        # Load configuration from config.json
        with open('config.json') as config_file:
            config = json.load(config_file)

        table_id = config['pointwise_eval_table']
        dataset_id = config['eval_dataset']
        project_id = config["project"]
        location_id=config["project_location"]
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        dataset_full_id = f"{project_id}.{dataset_id}"

        #remove unwanted characters from column name
        result.columns = result.columns.str.replace("/", "-")

        # Initialize BigQuery Client
        client = bigquery.Client()


        # Ensure the dataset exists
        try:
            client.get_dataset(dataset_full_id)
            print(f"Dataset {dataset_full_id} exists.")
        except NotFound:
            print(f"Dataset {dataset_full_id} not found. Creating dataset...")
            dataset = bigquery.Dataset(dataset_full_id)
            dataset.location = location_id 
            client.create_dataset(dataset)
            print(f"Dataset {dataset_full_id} created successfully.")




        try:
            # Fetch the existing table
            table = client.get_table(table_full_id)
            existing_schema = {field.name: field.field_type for field in table.schema}
            print(f"Table {table_full_id} exists. Checking schema...")

            # Infer schema from DataFrame
            new_schema = {
                name: bigquery.enums.SqlTypeNames.DATE if (dtype == 'object'  and name=='run_experiment_date')
                else bigquery.enums.SqlTypeNames.STRING if dtype == 'object'
                else bigquery.enums.SqlTypeNames.FLOAT if dtype in ['float64', 'float32']
                else bigquery.enums.SqlTypeNames.INTEGER if dtype in ['int64', 'int32']
                else bigquery.enums.SqlTypeNames.BOOLEAN if dtype == 'bool'
                else bigquery.enums.SqlTypeNames.TIMESTAMP if dtype == 'datetime64[ns]'
                else bigquery.enums.SqlTypeNames.STRING
                for name, dtype in zip(result.columns, result.dtypes)
            }

            # Identify schema differences
            schema_changes = []
            for col, dtype in new_schema.items():
                if col not in existing_schema:
                    # Add new column
                    schema_changes.append(bigquery.SchemaField(col, dtype))
                elif existing_schema[col] != dtype:
                    print(f"Type change detected for column '{col}' from {existing_schema[col]} to {dtype}.")
                    # BigQuery doesn't allow direct type changes; handle as needed.

            if schema_changes:
                print("Altering schema to add new columns...")
                table.schema = table.schema + schema_changes
                table = client.update_table(table, ["schema"])
                print(f"Table {table_full_id} schema updated successfully.")
            else:
                print("Schema is already up-to-date.")

        except NotFound:
            print(f"Table {table_full_id} not found. Creating table...")
            # Infer schema from DataFrame
            schema = [
                bigquery.SchemaField(name, bigquery.enums.SqlTypeNames.DATE if (dtype == 'object'  and name=='run_experiment_date')
                                     else bigquery.enums.SqlTypeNames.STRING if dtype == 'object' 
                                     else bigquery.enums.SqlTypeNames.FLOAT if dtype in ['float64', 'float32']
                                     else bigquery.enums.SqlTypeNames.INTEGER if dtype in ['int64', 'int32']
                                     else bigquery.enums.SqlTypeNames.BOOLEAN if dtype == 'bool'
                                     else bigquery.enums.SqlTypeNames.TIMESTAMP if dtype == 'datetime64[ns]'
                                     else bigquery.enums.SqlTypeNames.STRING)
                for name, dtype in zip(result.columns, result.dtypes)
            ]

            # Create the table
            table = bigquery.Table(table_full_id, schema=schema)
            table = client.create_table(table)
            print(f"Table {table_full_id} created successfully.")

        # Define job configuration
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        # Save DataFrame to BigQuery
        job = client.load_table_from_dataframe(result, table_full_id, job_config=job_config)
        job.result()  # Wait for the job to complete

        # Additional error inspection after the job completes
        if job.errors:
            print("The job completed with the following errors:")
            for error in job.errors:
                print(f" - {error['message']}")
        else:
            print(f"Evaluations have successfully been loaded into {table_full_id}.")

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


    def get_evaluations(self):
        """
        Extracts the evaluation metricsusing:
            1-user defined metrics and rating criteria
            2-pre-defined mathematical metrics: perplexity, entropy

        """
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
        
        #calcualte user defined metrics
        if self.eval_metrics:
            metrics=[]
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

    