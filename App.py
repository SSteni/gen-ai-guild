
import os
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, pipeline
import sqlite3
import openai
import autoplotlib as aplt

#%% App + Frontend + Autoplotlib

#Initialize Streamlit app
st.title("Prescription Data Analysis")
    
#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#The above for loop iterates through the chat history and displays each message in the chat message container 
#(with the author role and message content)

#React to user input
if query := st.chat_input("Hello! How can I help?"):    
        
    #Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
        
    #Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})    
    
    #Data Connection - Make connection with the db
    connection = sqlite3.connect('prescription.db')

    #my_table is the table name
    query_sql = "SELECT * FROM my_table"
    data = pd.read_sql_query(query_sql, connection)
    df = data.head(250)
        
    #Convert all columns to string type if they are not already.
    df = df.astype(str)

    #Create a dictionary from the DataFrame.
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

    #Initialize the TAPAS model and tokenizer.
    model_path = r"C:\Users\eljya\OneDrive\Documents\UST - LLM work\TAPEX_hf_model"
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model_path)
    tapas_tokenizer = AutoTokenizer.from_pretrained(model_path)

    #Create the TAPAS pipeline.
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    #Create a list of tables with the same query.
    tables_with_query = [{'table': table_data, 'query': query}]

    #Perform question answering.
    result = nlp(tables_with_query)

    #Extract the TAPAS model's answer from the result
    tapas_answer = result['answer']
    print(tapas_answer)
    
    #Define your OpenAI GPT-3 API key
    openai.api_key = "insert your openai api key here"

    #Define a prompt for GPT-3 using the TAPAS answer
    prompt = f"Query: {query}\nAnswer: {tapas_answer}\nResponse:"
    print("prompt:", prompt)
    
    #Use GPT-3 to generate a natural response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  #Choose the most appropriate GPT-3 engine
        prompt=prompt,
        max_tokens=4000,  #Adjust the max_tokens as needed for the response length
        temperature=0.7,  #Adjust the temperature for randomness in the response
    )

    #Extract the generated response
    generated_response = response.choices[0].text.strip()
    print(generated_response)
    
    #Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(generated_response)
        
        #Autoplotlib plot
        os.environ["OPENAI_API_KEY"] = openai.api_key
        code, fig, llm_response = aplt.plot(query, data=data)
        st.write(fig)
        
        
    #Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": generated_response})