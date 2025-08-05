# tasks/object_tasks.py
from crewai import Task

class AgentTasks:
    @staticmethod
    def create_metadata_task(bigquery_project_id: str, dataset_id: str, object_type: str, object_name: str, agent):
        return Task(
            description=f"Get metadata for project {bigquery_project_id}, dataset {dataset_id} and object `{object_name}` from the bigquery.",
            expected_output="Detailed column info including names and data types.",
            agent=agent,
            input={"bigquery_project_id": bigquery_project_id,"dataset_id":dataset_id, "object_type": object_type, "object_name": object_name}
        )
    
    @staticmethod
    def create_mapping_task(object_metadata: dict, agent):
        return Task(
            description=f"Convert the following object metadata to a BigQuery-compatible model as a experience data engineer who follows gcp best practices for data engineering:\n{object_metadata}",
            expected_output="generate a .sql file including bigquery statement",
            agent=agent
        )
