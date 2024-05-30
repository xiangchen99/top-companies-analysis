#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:16:59 2024

@author: xiangchen
"""


#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import csv
import requests
import torch
#%% Data scrape best buy website
url = "https://www.bestbuy.com/site/laptop-computers/all-laptops/pcmcat138500050001.c?id=pcmcat138500050001"

#this function applies a header to get a website
def extract_source(url):
    headers = {"User-Agent":"Mozilla/5.0"}
    source=requests.get(url, headers=headers).text
    return source
soup = BeautifulSoup(extract_source(url), "html.parser")
print(soup.prettify())

#%% try to extract each cell
item_list = soup.find("ol", class_ = "sku-item-list")
print(item_list)