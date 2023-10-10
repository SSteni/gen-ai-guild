import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, pipeline
import sqlite3
import openai

# Initialize Streamlit app
st.title("Prescription Data Analysis")

# Input box for user query
query = st.text_input("Enter your query:")
print(query)
# Button to trigger the analysis
if st.button("Analyze"):
    # Step 1. Data Connection
    # Make connection with the db
    connection = sqlite3.connect('prescription.db')

    # my_table is the table name
    query_sql = "SELECT * FROM my_table"
    data = pd.read_sql_query(query_sql, connection)
    df = data.head(250)
    # Convert all columns to string type if they are not already.
    df = df.astype(str)

    # Create a dictionary from the DataFrame.
    table_data = {
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

    # Initialize the TAPAS model and tokenizer.
    model_path = "/Users/ssteni/Documents/tapex/tapas_hf_download/"
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model_path)
    tapas_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create the TAPAS pipeline.
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    # Create a list of tables with the same query.
    tables_with_query = [{'table': table_data, 'query': query}]

    # Perform question answering.
    result = nlp(tables_with_query)

    # Extract the TAPAS model's answer from the result
    tapas_answer = result['answer']
    print(tapas_answer)
    # Define your OpenAI GPT-3 API key
    openai.api_key = "YOUR_API_KEY"

    # Define a prompt for GPT-3 using the TAPAS answer
    prompt = f"Query: {query}\nAnswer: {tapas_answer}\nResponse:"
    print("prompt:", prompt)
    # Use GPT-3 to generate a natural response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # Choose the most appropriate GPT-3 engine
        prompt=prompt,
        max_tokens=4000,  # Adjust the max_tokens as needed for the response length
        temperature=0.7,  # Adjust the temperature for randomness in the response
    )

    # Extract the generated response
    generated_response = response.choices[0].text.strip()
    print(generated_response)
    # Display the generated response
    st.write("Generated Response:")
    st.write(generated_response)
