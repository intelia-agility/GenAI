#from langchain.tools import tool
from sqlalchemy import create_engine, inspect
from crewai.tools import tool
from google.cloud import bigquery

# tools/object_metadata_tool.py
@tool
def get_object_metadata(bigquery_project_id: str, dataset_id: str, object_type: str, object_name: str) -> dict:
    """
    Given a BigQuery object type (table/view) and name, returns metadata (columns, types).
    """
   
    client = bigquery.Client(project=bigquery_project_id)
    table_ref = f"{bigquery_project_id}.{dataset_id}.{object_name}"


    try:
        table = client.get_table(table_ref)
        print(table)
        columns = [
            {"name": schema_field.name, "type": schema_field.field_type}
            for schema_field in table.schema
        ]        
        
        return {
            "object": object_name,
            "type": object_type,
            "columns": columns
        }
    except Exception as e:
        return {
            "object": object_name,
            "type": object_type,
            "error": str(e),
            "note": "Custom logic needed for procedures/functions if applicable."
        }
