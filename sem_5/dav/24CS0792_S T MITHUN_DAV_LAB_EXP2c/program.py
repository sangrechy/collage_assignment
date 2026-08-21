"""
AIM: Read data from text files, Excel spreadsheets, and web-based sources.
"""
import pandas as pd

text_df = pd.read_csv("../data/data.csv")
excel_df = pd.read_excel("../data/data.xlsx", sheet_name="Sheet1")
web_df = pd.read_csv("https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv")

print("Text file data:\n", text_df.head())
print("\nExcel file data:\n", excel_df.head())
print("\nWeb data:\n", web_df.head())

text_df = text_df.ffill()
excel_df = excel_df.bfill()
web_df = web_df.dropna()

text_df.to_csv("processed_text.csv", index=False)
excel_df.to_excel("processed_excel.xlsx", index=False)

print("\nSaved processed_text.csv and processed_excel.xlsx")
