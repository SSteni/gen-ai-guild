import pandas as pd
df = pd.read_csv('T201912PDPI BNFT.csv')

# Create a new column 'Explanation'
texts = []

for _, row in df.iterrows():
    text = (
        f"{row['SHA']} is the identifier for the healthcare organization or region. "
        f"{row['PCT']} is the Primary Care Trust responsible for healthcare services. "
        f"{row['PRACTICE']} is the specific medical practice or clinic. "
        f"{row['BNF CODE']} is the unique code for identifying a medication in the BNF. "
        f"{row['BNF NAME']} is the name of the medication corresponding to the BNF code. "
        f"{row['ITEMS']} is the number of prescriptions (items) for the medication. "
        f"{row['NIC']} is the Net Ingredient Cost of the medication (excluding VAT). "
        f"{row['ACT COST']} is the Actual cost incurred for providing the medication. "
        f"{row['QUANTITY']} is the Quantity of the medication prescribed (e.g., 500.0 milliliters). "
        f"{row['PERIOD']} is the Year and month during which the prescriptions were made (e.g., December 2019)."
    )
    texts.append(text)

df['Text'] = texts

# # Display the dataframe
# print(df)

# Save the DataFrame to a .csv file
df.to_csv('data_preprocessed.csv', index=False)