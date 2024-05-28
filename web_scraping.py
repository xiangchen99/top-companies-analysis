# We will do the web scraping here

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import csv
import requests


#%% Getting the Info for the df
url = "https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

table = soup.find_all("table")[1]
world_titles = table.find_all("th")


world_table_titles = [title.text.strip() for title in world_titles]

column = table.find_all("tr")

total_rows = []
for row in column[1:]:
    row_data = row.find_all("td")
    individual_row_data = [data.text.strip() for data in row_data]
    total_rows.append(individual_row_data)
    

#%% Putting everything into a dataframe
df = pd.DataFrame(total_rows, columns=world_table_titles)

#%%exporting into csv
df.to_csv(r'C:\Users\xiang\Documents\Data Science\web_scraping\HighestUSCompanyRevenue.csv', index=False)

#%%cleaning the dataframe
df["Employees"] = df["Employees"].str.replace(",", '')
df["Revenue (USD millions)"] = df["Revenue (USD millions)"].str.replace(",",'')
df["Revenue growth"] = df["Revenue growth"].str.replace("%", '').astype(float)/100
df[['City', 'State']] = df['Headquarters'].str.extract(r'^(.*?),\s*(\w+|\w+\s\w+)$')
df = df.drop(columns = ["Headquarters"])

