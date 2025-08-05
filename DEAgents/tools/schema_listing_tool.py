from crewai.tools import tool 
from sqlalchemy import create_engine, inspect
from google.cloud import bigquery
# tools/schema_listing_tool.py
@tool
def list_bigquery_objects(project_id: str) -> dict:
    """
    Lists all object types (tables, views, routines, models, etc.)
    across all datasets in a BigQuery project.
    """
    client = bigquery.Client(project=project_id)
    project_summary = {}

    datasets = client.list_datasets(project=project_id)

    for dataset in datasets:
        dataset_id = dataset.dataset_id
        full_dataset_id = f"{project_id}.{dataset_id}"
        project_summary[dataset_id] = {
            "tables": [],
            "views": [],
            "materialized_views": [],
            "external_tables": [],
            "snapshots": [],
            "procedures": [],
            "functions": [],
            "models": []
        }

        # List tables and classify them
        tables = client.list_tables(full_dataset_id)
        for table in tables:
            if table.table_type == "TABLE":
                project_summary[dataset_id]["tables"].append(table.table_id)
            elif table.table_type == "VIEW":
                project_summary[dataset_id]["views"].append(table.table_id)
            elif table.table_type == "MATERIALIZED_VIEW":
                project_summary[dataset_id]["materialized_views"].append(table.table_id)
            elif table.table_type == "EXTERNAL":
                project_summary[dataset_id]["external_tables"].append(table.table_id)
            elif table.table_type == "SNAPSHOT":
                project_summary[dataset_id]["snapshots"].append(table.table_id)

        # List routines (functions and procedures)
        routines_query = f"""
            SELECT routine_name, routine_type
            FROM `{full_dataset_id}.INFORMATION_SCHEMA.ROUTINES`
        """
        try:
            routines = client.query(routines_query).result()
            for row in routines:
                if row.routine_type == "PROCEDURE":
                    project_summary[dataset_id]["procedures"].append(row.routine_name)
                elif row.routine_type == "FUNCTION":
                    project_summary[dataset_id]["functions"].append(row.routine_name)
        except Exception as e:
            print(f"Skipping routines for dataset {dataset_id}: {e}")

        # # List ML models
        # models_query = f"""
        #     SELECT model_name
        #     FROM `{full_dataset_id}.INFORMATION_SCHEMA.MODELS`
        # """
        # try:
        #     models = client.query(models_query).result()
        #     for row in models:
        #         project_summary[dataset_id]["models"].append(row.model_name)
        # except Exception as e:
        #     print(f"Skipping models for dataset {dataset_id}: {e}")

    return project_summary