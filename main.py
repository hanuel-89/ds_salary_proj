# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:17:27 2021

@author: Hanuel
"""

import glassdoor_scrapper2 as gs
import pandas as pd

path = "C:/Users/Hanuel/Desktop/ds_salary_proj/chromedriver"

df = gs.get_jobs('data scientist', 15, False, path, 20)

df