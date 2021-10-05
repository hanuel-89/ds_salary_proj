# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:19:33 2021

@author: Hanuel
"""

import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': "application/json"}
#data_in = np.data_in.values.reshape(1,-1)
data = {'input': data_in}

r = requests.get(URL, headers=headers, json=data)

r.json()
