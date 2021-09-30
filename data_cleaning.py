# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:22:18 2021

@author: Hanuel
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# salary paring
# company name text only
# state field
# age of company
# parsing of job descriptions (python, etc)

# STEP 1: SALARY PARSING

# Create an hourly and employer_provided column
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0 )
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0 )


# Remove all '-1' from the Salary Estimate column
df = df[df['Salary Estimate'] != '-1']

# Remove text field "Glassdoor est." from Salary estimate
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

# Remove K and $ from salary
minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
# Remove 'per hour' and 'employer provided salary:' from minus_Kd
min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

# Create min and max salary columns

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0])) # Minimum salary
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1])) # Maximum salary
df['avg_salary'] = (df['min_salary']+df['max_salary'])/2

# STEP 2: COMPANY NAME TEXT

df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)

# STEP 3: STATE FIELD

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])

df.job_state.value_counts()

# STEP 4: Find if the job position is at the HQ
df['some_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

# STEP 5: COMPANY AGE
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2021 - x)

# STEP 6: PARSE THE JOB DESCRIPTION

## List the common tools for data analysis
#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

#r studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

#spark
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# Drop the unnamed column
df_out = df.drop(['Unnamed: 0'], axis = 1)

df_out.to_csv('salary_data_cleaned.csv', index=False)


