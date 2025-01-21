from google.cloud import aiplatform
import vertexai
import pandas as pd
import json

from vertexai.evaluation import (
    EvalTask, 
    PointwiseMetric,
    PointwiseMetricPromptTemplate
)
 
import uuid 
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from datetime import datetime

class PointWiseEvaluationClient:
    """Wrapper around Pointwise Evaluation Client."""

    def __init__(
        self,
        project: str=None,
        location: str = "us-central1",
        items: list[str] = None,
        response_llm_model: str= None,
        eval_metrics: list[dict] =None,
        experiment_name: str="pointwise-evaluation-experiment",
        evaluation_prompt: str="Evaluate the AI's contribution to a meaningful content generation",
        delete_experiment: bool= True
        

    ):
        
       
        self.location = location
        self.project = project     
        self.items =items
        self.eval_metrics=eval_metrics
        self.experiment_name=experiment_name
        self.evaluation_prompt=evaluation_prompt
        self.response_llm_model=response_llm_model
        self.delete_experiment=delete_experiment
        self.run_experiment_name=self.experiment_name+"-"+ str(uuid.uuid4())
    
        vertexai.init(project=self.project, location= self.location )
       
 
    def set_evaluation_data(self):
        """
        Sets the input data as in a dataframe for evaluation

        """
            
        eval_dataset= pd.DataFrame(
                                {
                                   # "instruction": instructions,
                                   # "context": contexts,
                                    "response": self.items,
                                    "response_llm_model":[self.response_llm_model]*len(self.items),
                                    "run_experiment_name":[self.run_experiment_name]*len(self.items),
                                    "run_experiment_date" : [datetime.today().date()]*len(self.items),

                                }
                            )
       
        eval_dataset['run_experiment_date'] = pd.to_datetime(eval_dataset['run_experiment_date']).dt.strftime('%Y-%m-%d')
        
        return eval_dataset

    def log_evaluations(self, result):
        """
        Log the evaluation result into BigQuery, altering the table schema if needed.

        Args:
            result (dataframe): The evaluation result to be recorded into the database.
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


       
    def get_evaluations(self):
        """
        Get the evaluations using user defined rating criteria.

        """
        
        metrics=[]
        # Define  pointwise quality metric(s)
        for metric in self.eval_metrics:
            # Define a pointwise quality metric
            pointwise_quality_metric_prompt = f"""{self.evaluation_prompt}; evaluate {metric['metric']}.
            Rate the response on a 1-5 scale, using this rubric criteria:

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
       
        # set evaluation data
        eval_dataset=self.set_evaluation_data()
        
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
        
        #record the result into bigquery
        self.log_evaluations(results.metrics_table)
        
        #Delete the experiment after getting the result
        if self.delete_experiment:
            experiment = aiplatform.Experiment(self.experiment_name)
            experiment.delete()
 
        return results.metrics_table

    