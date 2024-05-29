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

#%%
print(df.dtypes)
#%% searching for high population
#high population is when the employees are above 1 million
high_pop_df = df[df["Employees"] > 1000000]


#%% pytorch test
import math

dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
 

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')