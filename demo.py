import pandas as pd
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, pipeline
import sqlite3
from huggingface_hub import hf_hub_download
import os
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, pipeline

# Step 1. Data Connection
# Make connection with the db
connection = sqlite3.connect('prescription.db')

# my_table is the table name
query = "SELECT * FROM my_table"
data = pd.read_sql_query(query, connection)

# print(df.head())

df = data.head(250)
# connection.close()

# # Model Connection
# HUGGING_FACE_API_KEY = "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"

# # Replace this if you want to use a different model
# model_id = "navteca/tapas-large-finetuned-wtq"
# filenames = [
#     "pytorch_model.bin", "config.json",
#     "special_tokens_map.json", "vocab.txt", "tokenizer_config.json"
# ]

# Set the destination directory to the current folder (the dot . represents the current folder)
# destination_directory = "/Users/ssteni/.cache/huggingface/hub/models--navteca--tapas-large-finetuned-wtq/snapshots/cd7feb8b379e08187f8927debec340fa05ca3715/"
# os.makedirs(destination_directory, exist_ok=True)
# # 	I worked out the filenames by browsing Files and versions on the Hugging Face UI.

# for filename in filenames:
#     downloaded_model_path = hf_hub_download(
#         repo_id=model_id,
#         filename=filename,
#         token=HUGGING_FACE_API_KEY
#     )
#     # destination_path = os.path.join(destination_directory, filename)
#     # os.rename(downloaded_model_path, destination_path)

# print(downloaded_model_path)
# # print(destination_path)

# Inference
# Convert all columns to string type if they are not already.
df = df.astype(str)
print(df.columns)
# print(df.head())

# Create a dictionary from the DataFrame.
table_data = {
    # 'Unnamed: 0': df['Unnamed: 0'].tolist(),
    'SHA': df['SHA'].tolist(),
    'PCT': df['PCT'].tolist(),
    'PRACTICE': df['PRACTICE'].tolist(),
    'BNF CODE': df['BNF CODE'].tolist(),
    'BNF NAME': df['BNF NAME'].tolist(),
    'ITEMS': df['ITEMS'].tolist(),
    'NIC': df['NIC'].tolist(),
    'ACT COST': df['ACT COST'].tolist(),
    'QUANTITY': df['QUANTITY'].tolist(),
    'PERIOD': df['PERIOD'].tolist(),
}

# model_path = "/Users/ssteni/.cache/huggingface/hub/models--navteca--tapas-large-finetuned-wtq/snapshots/cd7feb8b379e08187f8927debec340fa05ca3715/"
model_path = "/Users/ssteni/Documents/tapex/tapas_hf_download/"
# Initialize the TAPAS model and tokenizer.
tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model_path)
tapas_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define your query here.
# query = "What are the top five BNF names (BNF NAME) by the number of items prescribed?"
# query = "Which practice (PRACTICE code) prescribed the most items?"
# query = "What is the total net ingredient cost (NIC) for each BNF code?"
# query = "Which medication has the highest average cost per item?"
# query = "Which SHA (SHA code) prescribed the most items?"
# query = "What is the total NIC (Net Ingredient Cost) for PRACTICE N81002?"
# query = "Find the BNF NAME for BNF CODE 0101021B0BEADAJ."
# query = "How many items were prescribed in December 2019?"
# query = "List the top 3 PRACTICE codes by the number of items prescribed."
# query = "What is the average ACT COST per item for PRACTICE N81002?"
query = "Find the QUANTITY for BNF CODE 0101021B0AAALAL."

# Create the TAPAS pipeline.
nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

# Create a list of tables with the same query.
tables_with_query = [{'table': table_data, 'query': query}]

# Perform question answering.
result = nlp(tables_with_query)

# Print the result.
print(result)
