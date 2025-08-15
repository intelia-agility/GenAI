#from langchain.tools import tool
from sqlalchemy import create_engine, inspect
from crewai.tools import tool
from google.cloud import bigquery

from typing import Dict, Any, Optional
from google.cloud import bigquery
from langchain.tools import tool
import json


# @tool("get_object_metadata")
# def get_object_metadata(
#     project_id: str,
#     dataset_id: str,
#     table_id: str,
#     include_sample_values: Optional[bool] = False,
#     sample_row_limit: Optional[int] = 5
# ) -> Dict[str, Any]:
#     """
#     Retrieves comprehensive metadata for a BigQuery object (table or view), including:
#     - table structure, partitioning, clustering, view definition
#     - column metadata and nested field structures
#     - optional: sample values and nested stats
#     - IAM policy bindings

#     Args:
#         project_id: The GCP project ID.
#         dataset_id: The BigQuery dataset ID.
#         table_id: The BigQuery table or view name.
#         include_sample_values: Whether to query sample values.
#         sample_row_limit: Max number of rows to fetch if sampling is enabled.

#     Returns:
#         A dictionary with full metadata.
#     """
#     client = bigquery.Client(project=project_id)
#     full_table_id = f"{project_id}.{dataset_id}.{table_id}"

#     try:
#         table = client.get_table(full_table_id)
#     except Exception as e:
#         return {"error": f"Could not fetch table: {e}"}

#     # Table-level metadata
#     table_metadata = {
#         "name": table.table_id,
#         "project_id": table.project,
#         "dataset_id": table.dataset_id,
#         "table_type": table.table_type,
#         "description": table.description,
#         "labels": table.labels,
#         "num_rows": table.num_rows,
#         "size_bytes": table.num_bytes,
#         "creation_time": table.created.isoformat() if table.created else None,
#         "last_modified_time": table.modified.isoformat() if table.modified else None,
#         "expiration_time": table.expires.isoformat() if table.expires else None,
#     }

#     # Partitioning & clustering
#     if table.time_partitioning:
#         table_metadata["partitioning"] = {
#             "type": table.time_partitioning.type_,
#             "field": table.time_partitioning.field,
#             "require_partition_filter": table.time_partitioning.require_partition_filter,
#         }

#     if table.range_partitioning:
#         table_metadata["range_partitioning"] = {
#             "field": table.range_partitioning.field,
#             "range": {
#                 "start": table.range_partitioning.range_.start,
#                 "end": table.range_partitioning.range_.end,
#                 "interval": table.range_partitioning.range_.interval,
#             },
#         }

#     if table.clustering_fields:
#         table_metadata["clustering"] = table.clustering_fields

#     # View info
#     if table.table_type == "VIEW":
#         table_metadata["view_definition"] = table.view_query
#         table_metadata["use_legacy_sql"] = table.view_use_legacy_sql

#     # Column-level metadata
#     columns = []
#     for field in table.schema:
#         col = {
#             "name": field.name,
#             "type": field.field_type,
#             "mode": field.mode,
#             "description": field.description,
#         }
#         if field.field_type in ["RECORD", "STRUCT"]:
#             col["subfields"] = [
#                 {
#                     "name": subfield.name,
#                     "type": subfield.field_type,
#                     "mode": subfield.mode,
#                     "description": subfield.description
#                 } for subfield in field.fields
#             ]
#         columns.append(col)

#     # Nested column profiling from INFORMATION_SCHEMA
#     try:
#         profile_query = f"""
#         SELECT column_path, data_type, is_nullable, 
#                MIN(ordinal_position) AS ordinal_position
#         FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
#         WHERE table_name = '{table_id}'
#         GROUP BY column_path, data_type, is_nullable
#         ORDER BY ordinal_position
#         """
#         results = client.query(profile_query).result()
#         nested_column_info = [
#             {
#                 "column_path": row.column_path,
#                 "data_type": row.data_type,
#                 "nullable": row.is_nullable
#             } for row in results
#         ]
#     except Exception as e:
#         nested_column_info = [{"error": f"Failed nested stats: {e}"}]

#     # Optional sample values
#     sample_values = []
#     if include_sample_values:
#         try:
#             query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {sample_row_limit}"
#             sample = client.query(query).result()
#             for row in sample:
#                 sample_values.append(dict(row.items()))
#         except Exception as e:
#             sample_values = [{"error": f"Sample fetch failed: {e}"}]

#     # Access control (IAM policy)
#     try:
#         policy = client.get_iam_policy(full_table_id)
#         iam_bindings = [{"role": b.role, "members": list(b.members)} for b in policy.bindings]
#     except Exception as e:
#         iam_bindings = [{"error": f"Could not fetch IAM policy: {e}"}]

#     return {
#         "table": table_metadata,
#         "columns": columns,
#         "nested_column_stats": nested_column_info,
#         "sample_values": sample_values if include_sample_values else "Skipped",
#         "access_control": iam_bindings
#     }


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
