import pandas as pd
import sqlite3


df = pd.read_csv('T201912PDPI BNFT.csv')

# Connect to the SQLite database (creates a new one if not exists)
conn = sqlite3.connect('prescription.db')

# Write the DataFrame to the database
df.to_sql('my_table', conn, if_exists='replace', index=False)

# Close the database connection
conn.close()
