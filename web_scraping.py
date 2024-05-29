# We will do the web scraping here

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import csv
import requests
import torch

#%% set torch device for macos
device = torch.device("mps")

#%% set torch device for cuda
device = torch.device("cuda")

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
#df.to_csv(r'C:\Users\xiang\Documents\Data Science\web_scraping\HighestUSCompanyRevenue.csv', index=False)

#%%cleaning the dataframe
df["Employees"] = df["Employees"].str.replace(",", '')
df["Revenue (USD millions)"] = df["Revenue (USD millions)"].str.replace(",",'')
df["Revenue growth"] = df["Revenue growth"].str.replace("%", '').astype(float)/100
df[['City', 'State']] = df['Headquarters'].str.extract(r'^(.*?),\s*(\w+|\w+\s\w+)$')
df = df.drop(columns = ["Headquarters"])

#%% Converting the dtypes
# Dictionary specifying the conversion types for each column
conversion_types = {
    'Rank': 'string',
    'Name': 'string',
    'Industry': 'string',
    'Revenue (USD millions)': float,
    'Revenue growth': float,
    'City': 'string',
    'State': 'string'
}

# Converting data types
df = df.astype(conversion_types)

df['Employees'] = df['Employees'].str.replace(r'\D', '', regex=True)
df['Employees'] = df['Employees'].astype('Int64')

df.dropna(inplace = True)

#%% searching for high population
#high population is when the employees are above 1 million
high_pop_df = df[df["Employees"] > 1000000]
print(high_pop_df)

#%% searching for small top companies

low_pop_df = df[df["Employees"] < 50000]
print(low_pop_df)

#%% We will try to determine what states have the highest concentration of companies
print(df["State"].nunique())

state_counts = df.groupby("State")["Name"].count()
state_counts_sorted = state_counts.sort_values(ascending=False)
print(state_counts_sorted)

# Plotting the results
state_counts_sorted.plot(kind='bar', color='skyblue')
plt.xlabel('State')
plt.ylabel('Number of Companies')
plt.title('Concentration of Companies by State')
plt.show()

#%% We will find the distribution of companies by top 10 biggest industry
# Counting companies in each industry
industry_counts = df['Industry'].value_counts()[:10]

plt.figure(figsize=(12, 8))  # Increased figure size
sns.barplot(x=industry_counts.index, y=industry_counts.values, palette='viridis')
plt.title('Distribution of Companies by Industry')
plt.ylabel('Number of Companies')
plt.xlabel('Industry')
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
plt.show()

#%% Finding a correlation between Revenue, Revenue Growth, and Employees
# Correlation matrix
correlation_data = df[['Revenue (USD millions)', 'Revenue growth', 'Employees']].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

## This is a bigger word (I guess)
#%% 