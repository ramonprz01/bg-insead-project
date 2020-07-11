# Approach 2 (Specific Companies Only)

**Idea:** Get a list of O*NET occupations and their associated job titles (there are over 800 occupations and at least 5-10 titles for each). Get humans to assign each job title within and then across occupations to its likely hierarchical level in a firm, regardless of which firm it’s in. Use continuous bag of words model to expand the job titles in O*NET to the full job title list in burning glass. Once done, we can simply assign levels to each job title in BG based on coder opinion. Then we count the number of unique levels in the firm and consider firms with a larger number of unique levels as ‘more hierarchical’.

**[Get the notebook here](https://nbviewer.jupyter.org/github/ramonprz01/bg-insead-project/blob/master/analysis/approach_2/code/analysis_approach2.ipynb)**

**Assumptions to test:**

Distribution of job titles within occupations: how many job titles correspond to each O*NET code from the BG data (occupation == O*NET code)  
a. For each firm, the number of occupations (unique) (BGTOcc)  
b. For every company-BGTOcc, unique number of job titles & list of those job titles. I reinterpreted this one as “For every company and every BGTOcc within it, get unique number of job titles & list of those job titles”  
c. For every job title, number of occupations it appears under  
d. Not just the # of jobs and # occupations but ideally the # of job titles matched to an occupation. How many unique job titles do you see in an occupation?  

**Additions from meeting on Friday, June 19**
1. List of job titles (consolidated & clean) e.g., for Facebook, which appear less than median (I think it was 2) times. [Link to answer](https://www.dropbox.com/s/ekqjjb4t00l8b74/fb_job_var_nums.xlsx?dl=0)
2. For a sample firm, xls with list of unique occupations (we need to read those and see whether they  make sense in terms of functions / divisions)
[Link to answer](https://www.dropbox.com/s/ccll89c1vhqf6av/sample_firm_unique_occu.csv?dl=0)
3. Avg number of occupations a job title (both consolidated & clean) is assigned to (for a sample firm and across firms in the sample) [Link to answer](https://www.dropbox.com/sh/w2j41ka4hsgx0gf/AAAdEGOsjI9_gK8__ZeVHE5Ta?dl=0)
    - For each job title in a company, how many different occupations this job map into?
    - For each of the 20 most common canon job titles, extract all unique clean job titles that correspond to them
4. Xls with examples of those job titles assigned to multiple occupations: is there an occupation they are primarily associated to, and the others are just some noise


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import jit, njit, vectorize
from glob import glob
import re
import seaborn as sns
import pickle
import joblib
import nltk
from collections import Counter

import concurrent.futures as cf

pd.set_option('display.max_columns', None) 

%matplotlib inline
```


```python
dtypes={'JobID': np.str, 'CleanJobTitle': np.str, 'JobDomain': np.str, 
        'CanonCity': np.str, 'CanonCountry': np.str, 'CanonState': np.str, 
        'JobText': np.str, 'JobURL': np.str, 'PostingHTML': np.float64, 
        'Source': np.str, 'JobReferenceID': np.str, 'Email': np.str, 
        'CanonEmployer': np.str, 'Latitude': np.str, 'Longitude': np.str, 
        'CanonIntermediary': np.str, 'Telephone': np.str, 'CanonJobTitle': 'object', 
        'CanonCounty': np.str, 'DivisionCode': np.float64, 'MSA': np.str, 'LMA': np.str,
        'InternshipFlag': np.str, 'ConsolidatedONET': np.str, 'CanonCertification': np.str, 
        'CanonSkillClusters': np.str, 'CanonSkills': np.str, 'IsDuplicate': np.str, 
        'IsDuplicateOf': np.float64, 'CanonMaximumDegree': np.str, 'CanonMinimumDegree': np.str, 
        'CanonOtherDegrees': np.str, 'CanonPreferredDegrees': np.str,
        'CanonRequiredDegrees': np.str, 'CIPCode': np.str, 'StandardMajor': np.str, 
        'MaxExperience': np.float64, 'MinExperience': np.float64, 'ConsolidatedInferredNAICS': np.float64, 
        'BGTOcc': np.str, 'MaxAnnualSalary': np.float64, 'MaxHourlySalary': np.float64, 
        'MinAnnualSalary': np.float64, 'MinHourlySalary': np.float64, 'YearsOfExperience': np.str, 
        'CanonJobHours': np.str, 'CanonJobType': np.str, 'CanonPostalCode': np.str, 
        'CanonYearsOfExperienceCanonLevel': np.str, 'CanonYearsOfExperienceLevel': np.str, 
        'ConsolidatedTitle': np.str, 'Language': np.str, 'BGTSubOcc': np.str, 'JobDate': np.str,
        'ConsolidatedDegreeLevels': np.str, 'MaxDegreeLevel': np.float64, 'MinDegreeLevel': np.float64
                       }
```

Depending on your use case for this particular approach, you can use either of the following two lists of variables. The rest have been removed for the simple reason that they have either >90% missing or their field does not add any useful value to the dataset. For example, `PostingHTML` contains links to websites which have mostly been despricated by now.


```python
best_list = ['JobID', 'CleanJobTitle', 'CanonCity', 'CanonState', 'JobDate', 'JobText', 'Source', 'CanonEmployer',
             'Latitude', 'Longitude', 'CanonIntermediary', 'CanonJobTitle', 'CanonCounty', 'DivisionCode', 'MSA', 'LMA',
             'InternshipFlag', 'ConsolidatedONET', 'CanonSkillClusters', 'CanonSkills', 'IsDuplicate', 'CanonMinimumDegree', 
             'CanonRequiredDegrees', 'CIPCode', 'MinExperience', 'ConsolidatedInferredNAICS', 'BGTOcc', 'MaxAnnualSalary',
             'MaxHourlySalary', 'MinAnnualSalary', 'MinHourlySalary', 'YearsOfExperience', 'CanonJobHours', 'CanonJobType',
             'CanonPostalCode', 'CanonYearsOfExperienceCanonLevel', 'CanonYearsOfExperienceLevel', 'ConsolidatedTitle', 
             'Language', 'BGTSubOcc', 'ConsolidatedDegreeLevels', 'MaxDegreeLevel', 'MinDegreeLevel']
```

This is the version used for `Approach 2`.


```python
small_list = ['JobID', 'CleanJobTitle', 'JobDate', 'JobText', 'CanonEmployer', 'CanonCity', 'CanonState', 'Latitude', 'Longitude', 'CanonCounty', 
              'CanonJobTitle', 'DivisionCode', 'MSA', 'LMA', 'ConsolidatedONET', 'CanonSkillClusters', 
              'CanonSkills', 'MinExperience', 'ConsolidatedInferredNAICS', 'BGTOcc', 'CanonJobHours', 
              'CanonJobType', 'CanonYearsOfExperienceLevel', 'ConsolidatedTitle', 'BGTSubOcc'
             ]
```

## 1. The Data -- For multiple files use the following code


```python
# Add your path a the wildcard * to select multiple files with glob

files = glob('/Users/ramonperez/Dropbox/Burning Glass/Analysis/company_data/x_co*.csv') 
files[:3]
```




    ['/Users/ramonperez/Dropbox/Burning Glass/Analysis/company_data/x_companies_2.csv',
     '/Users/ramonperez/Dropbox/Burning Glass/Analysis/company_data/x_companies_3.csv',
     '/Users/ramonperez/Dropbox/Burning Glass/Analysis/company_data/x_companies_1.csv']



#### One approach is to read the data with a list comprehension

This is fine for 1 to 3 GB of data that fits into memory.


```python
%%time

# Create a list containing all of the datasets
dfs = [pd.read_csv(f, low_memory=False, parse_dates=['JobDate'], usecols=small_list, dtype=dtypes) for f in files]

# Concatenate all of the datasets into one
df = pd.concat(dfs)
df.reset_index(drop=True, inplace=True)
df.head(3)
```

    CPU times: user 28.5 s, sys: 27.2 s, total: 55.7 s
    Wall time: 1min 52s


#### The other approach is to take advantage of multiprocessing

This is great for 2 or more GB of data. Keep in mind that the size of the data you will read should be less than your RAM.


```python
%%time


def get_data(data):
    return pd.read_csv(data, low_memory=False,
                       parse_dates=['JobDate'], 
                       usecols=small_list, dtype=dtypes)

with cf.ProcessPoolExecutor() as executor:
    results = executor.map(get_data, files)
    
df = pd.concat(results)
df.reset_index(drop=True, inplace=True)
df.head(3)
```

    CPU times: user 9.24 s, sys: 39.6 s, total: 48.8 s
    Wall time: 3min 53s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGTOcc</th>
      <th>BGTSubOcc</th>
      <th>CanonCity</th>
      <th>CanonCounty</th>
      <th>CanonEmployer</th>
      <th>CanonJobHours</th>
      <th>CanonJobTitle</th>
      <th>CanonJobType</th>
      <th>CanonSkillClusters</th>
      <th>CanonSkills</th>
      <th>CanonState</th>
      <th>CanonYearsOfExperienceLevel</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
      <th>ConsolidatedTitle</th>
      <th>DivisionCode</th>
      <th>JobDate</th>
      <th>JobID</th>
      <th>JobText</th>
      <th>LMA</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MSA</th>
      <th>MinExperience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Sp...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-25</td>
      <td>38201188919</td>
      <td>Sales Associates\n-Foley,AL36535\n-3781 S McKe...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>38200222343</td>
      <td>AUTO SALES ASSOCIATES, ENTRY LEVEL\n\nGULF CHR...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53-7061.00</td>
      <td>Detailer</td>
      <td>Fowlerville</td>
      <td>Livingston</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Automotive Detailer</td>
      <td>permanent</td>
      <td>Specialized Skills|Specialized Skills</td>
      <td>{'Automotive Services Industry Knowledge': 'Sp...</td>
      <td>MI</td>
      <td>NaN</td>
      <td>Automotive Service Detailer</td>
      <td>336111.0</td>
      <td>53706100</td>
      <td>Automotive Detailer</td>
      <td>47664.0</td>
      <td>2017-03-20</td>
      <td>38198645467</td>
      <td>Automotive Service Detailer\n\nChrysler Dealer...</td>
      <td>DV264764|MT261982</td>
      <td>42.6645</td>
      <td>-84.0695</td>
      <td>19820: Metropolitan Statistical Area|220: Comb...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGTOcc</th>
      <th>BGTSubOcc</th>
      <th>CanonCity</th>
      <th>CanonCounty</th>
      <th>CanonEmployer</th>
      <th>CanonJobHours</th>
      <th>CanonJobTitle</th>
      <th>CanonJobType</th>
      <th>CanonSkillClusters</th>
      <th>CanonSkills</th>
      <th>CanonState</th>
      <th>CanonYearsOfExperienceLevel</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
      <th>ConsolidatedTitle</th>
      <th>DivisionCode</th>
      <th>JobDate</th>
      <th>JobID</th>
      <th>JobText</th>
      <th>LMA</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MSA</th>
      <th>MinExperience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Sp...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-25</td>
      <td>38201188919</td>
      <td>Sales Associates\n-Foley,AL36535\n-3781 S McKe...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>38200222343</td>
      <td>AUTO SALES ASSOCIATES, ENTRY LEVEL\n\nGULF CHR...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53-7061.00</td>
      <td>Detailer</td>
      <td>Fowlerville</td>
      <td>Livingston</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Automotive Detailer</td>
      <td>permanent</td>
      <td>Specialized Skills|Specialized Skills</td>
      <td>{'Automotive Services Industry Knowledge': 'Sp...</td>
      <td>MI</td>
      <td>NaN</td>
      <td>Automotive Service Detailer</td>
      <td>336111.0</td>
      <td>53706100</td>
      <td>Automotive Detailer</td>
      <td>47664.0</td>
      <td>2017-03-20</td>
      <td>38198645467</td>
      <td>Automotive Service Detailer\n\nChrysler Dealer...</td>
      <td>DV264764|MT261982</td>
      <td>42.6645</td>
      <td>-84.0695</td>
      <td>19820: Metropolitan Statistical Area|220: Comb...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (718629, 25)



We will normalise the Employer names a bit in this dataset in a manual way by mapping the clean names to the not so clean ones. **NB**: This would not be a good approach to the larger dataset.


```python
dict_comps = {
    'Chrysler':'Chrysler',
    'Ford Motor Company':'Ford Motor Company', 
    'JP Morgan Chase Company':'JP Morgan Chase Company', 
    'McKinsey & Company':'McKinsey & Company',
    'Boston Consulting Group Incorporated':'Boston Consulting Group',
    'Boston Consulting Group':'Boston Consulting Group',
    'Bain Company':'Bain Company',
    'Bain Company Incorporated':'Bain Company',
    'General Motors':'General Motors',
    'Microsoft Corporation':'Microsoft Corporation',
    'Citi':'Citi',
    'PepsiCo Inc.':'PepsiCo Inc.',
    'Tyson Foods Incorporated':'Tyson Foods Incorporated',
    'Nestle USA Incorporated':'Nestle USA Incorporated',
    'Bank of America':'Bank of America',
    'The Goldman Sachs Group, Inc.':'The Goldman Sachs Group, Inc.',
    'Morgan Stanley':'Morgan Stanley',
    'Kraft Foods':'Kraft Foods',
    'Anheuser-Busch Companies, Inc.':'Anheuser-Busch Companies, Inc.',
    'Google Inc.':'Google Inc.',
    'Facebook':'Facebook',
    'Twitter':'Twitter',
    'Yahoo':'Yahoo',
    'Chryslerdealer':'Chrysler',
    'Oliver Wyman':'Oliver Wyman',
    'Oliver Wyman, Inc':'Oliver Wyman',
    'Bank Of America Charlotte':'Bank of America',
    'Roland Berger':'Roland Berger',
    'Roland Berger Llc':'Roland Berger',
    'General Motorsgeneral Motors':'General Motors',
    'Facebookadvertisers':'Facebook'
}
```


```python
df['CanonEmployer'] = df['CanonEmployer'].map(dict_comps)
```

Let's now take out any observation without a job description.


```python
%%time

df = df[df['JobText'].notna()].copy()
```

    CPU times: user 2.27 s, sys: 8.86 s, total: 11.1 s
    Wall time: 20.8 s


Check out the space that is currently occupied in your computer's memory with the following command.


```python
df.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 718301 entries, 0 to 718628
    Data columns (total 25 columns):
     #   Column                       Non-Null Count   Dtype         
    ---  ------                       --------------   -----         
     0   BGTOcc                       645891 non-null  object        
     1   BGTSubOcc                    645870 non-null  object        
     2   CanonCity                    711832 non-null  object        
     3   CanonCounty                  709999 non-null  object        
     4   CanonEmployer                718301 non-null  object        
     5   CanonJobHours                396003 non-null  object        
     6   CanonJobTitle                410256 non-null  object        
     7   CanonJobType                 397617 non-null  object        
     8   CanonSkillClusters           707773 non-null  object        
     9   CanonSkills                  718301 non-null  object        
     10  CanonState                   718283 non-null  object        
     11  CanonYearsOfExperienceLevel  456126 non-null  object        
     12  CleanJobTitle                718211 non-null  object        
     13  ConsolidatedInferredNAICS    713747 non-null  float64       
     14  ConsolidatedONET             677157 non-null  object        
     15  ConsolidatedTitle            718261 non-null  object        
     16  DivisionCode                 383782 non-null  float64       
     17  JobDate                      718301 non-null  datetime64[ns]
     18  JobID                        718301 non-null  object        
     19  JobText                      718301 non-null  object        
     20  LMA                          707613 non-null  object        
     21  Latitude                     711850 non-null  object        
     22  Longitude                    711850 non-null  object        
     23  MSA                          699314 non-null  object        
     24  MinExperience                451429 non-null  float64       
    dtypes: datetime64[ns](1), float64(3), object(21)
    memory usage: 4.4 GB



```python
missing_vals = df.isnull().sum() / df.shape[0] * 100
missing_vals
```




    BGTOcc                         10.080732
    BGTSubOcc                      10.083656
    CanonCity                       0.900597
    CanonCounty                     1.155783
    CanonEmployer                   0.000000
    CanonJobHours                  44.869491
    CanonJobTitle                  42.885225
    CanonJobType                   44.644794
    CanonSkillClusters              1.465681
    CanonSkills                     0.000000
    CanonState                      0.002506
    CanonYearsOfExperienceLevel    36.499323
    CleanJobTitle                   0.012530
    ConsolidatedInferredNAICS       0.633996
    ConsolidatedONET                5.727961
    ConsolidatedTitle               0.005569
    DivisionCode                   46.570867
    JobDate                         0.000000
    JobID                           0.000000
    JobText                         0.000000
    LMA                             1.487956
    Latitude                        0.898091
    Longitude                       0.898091
    MSA                             2.643321
    MinExperience                  37.153227
    dtype: float64



Notice that some vars above have a high percentage of missing values. If you would like to get rid of some of them given a threshold, change the value 70 below to your desired threshold.


```python
to_drop = (missing_vals[missing_vals > 70]).index
df.drop(to_drop, axis=1, inplace=True)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGTOcc</th>
      <th>BGTSubOcc</th>
      <th>CanonCity</th>
      <th>CanonCounty</th>
      <th>CanonEmployer</th>
      <th>CanonJobHours</th>
      <th>CanonJobTitle</th>
      <th>CanonJobType</th>
      <th>CanonSkillClusters</th>
      <th>CanonSkills</th>
      <th>CanonState</th>
      <th>CanonYearsOfExperienceLevel</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
      <th>ConsolidatedTitle</th>
      <th>DivisionCode</th>
      <th>JobDate</th>
      <th>JobID</th>
      <th>JobText</th>
      <th>LMA</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MSA</th>
      <th>MinExperience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Sp...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-25</td>
      <td>38201188919</td>
      <td>Sales Associates\n-Foley,AL36535\n-3781 S McKe...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41-2031.00</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>38200222343</td>
      <td>AUTO SALES ASSOCIATES, ENTRY LEVEL\n\nGULF CHR...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53-7061.00</td>
      <td>Detailer</td>
      <td>Fowlerville</td>
      <td>Livingston</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Automotive Detailer</td>
      <td>permanent</td>
      <td>Specialized Skills|Specialized Skills</td>
      <td>{'Automotive Services Industry Knowledge': 'Sp...</td>
      <td>MI</td>
      <td>NaN</td>
      <td>Automotive Service Detailer</td>
      <td>336111.0</td>
      <td>53706100</td>
      <td>Automotive Detailer</td>
      <td>47664.0</td>
      <td>2017-03-20</td>
      <td>38198645467</td>
      <td>Automotive Service Detailer\n\nChrysler Dealer...</td>
      <td>DV264764|MT261982</td>
      <td>42.6645</td>
      <td>-84.0695</td>
      <td>19820: Metropolitan Statistical Area|220: Comb...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Let's inspect the amount of occupations found in each company before moving to the next section.


```python
companies = df.groupby(['CanonEmployer'])
```


```python
companies[['BGTOcc','ConsolidatedInferredNAICS', 'ConsolidatedONET']].agg(['count'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>BGTOcc</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>CanonEmployer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Anheuser-Busch Companies, Inc.</th>
      <td>4298</td>
      <td>5005</td>
      <td>4703</td>
    </tr>
    <tr>
      <th>Bain Company</th>
      <td>1013</td>
      <td>520</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>Bank of America</th>
      <td>128493</td>
      <td>147389</td>
      <td>138164</td>
    </tr>
    <tr>
      <th>Boston Consulting Group</th>
      <td>1765</td>
      <td>767</td>
      <td>2030</td>
    </tr>
    <tr>
      <th>Chrysler</th>
      <td>48776</td>
      <td>50877</td>
      <td>50150</td>
    </tr>
    <tr>
      <th>Citi</th>
      <td>42399</td>
      <td>47884</td>
      <td>44461</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>23208</td>
      <td>25818</td>
      <td>24491</td>
    </tr>
    <tr>
      <th>Ford Motor Company</th>
      <td>9564</td>
      <td>10628</td>
      <td>9986</td>
    </tr>
    <tr>
      <th>General Motors</th>
      <td>28492</td>
      <td>30174</td>
      <td>29274</td>
    </tr>
    <tr>
      <th>Google Inc.</th>
      <td>22693</td>
      <td>25117</td>
      <td>23769</td>
    </tr>
    <tr>
      <th>JP Morgan Chase Company</th>
      <td>166179</td>
      <td>180439</td>
      <td>170367</td>
    </tr>
    <tr>
      <th>Kraft Foods</th>
      <td>2834</td>
      <td>3172</td>
      <td>3029</td>
    </tr>
    <tr>
      <th>McKinsey &amp; Company</th>
      <td>6363</td>
      <td>7981</td>
      <td>6830</td>
    </tr>
    <tr>
      <th>Microsoft Corporation</th>
      <td>34324</td>
      <td>38960</td>
      <td>38581</td>
    </tr>
    <tr>
      <th>Morgan Stanley</th>
      <td>14809</td>
      <td>17865</td>
      <td>15240</td>
    </tr>
    <tr>
      <th>Nestle USA Incorporated</th>
      <td>21059</td>
      <td>22612</td>
      <td>21773</td>
    </tr>
    <tr>
      <th>Oliver Wyman</th>
      <td>109</td>
      <td>42</td>
      <td>114</td>
    </tr>
    <tr>
      <th>PepsiCo Inc.</th>
      <td>48220</td>
      <td>50709</td>
      <td>49435</td>
    </tr>
    <tr>
      <th>Roland Berger</th>
      <td>22</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>The Goldman Sachs Group, Inc.</th>
      <td>15073</td>
      <td>18320</td>
      <td>15774</td>
    </tr>
    <tr>
      <th>Twitter</th>
      <td>3980</td>
      <td>4116</td>
      <td>4106</td>
    </tr>
    <tr>
      <th>Tyson Foods Incorporated</th>
      <td>18866</td>
      <td>21590</td>
      <td>20326</td>
    </tr>
    <tr>
      <th>Yahoo</th>
      <td>3352</td>
      <td>3761</td>
      <td>3475</td>
    </tr>
  </tbody>
</table>
</div>



The `BTGOcc` var contains values with a `-` in them so we will filtered `NaN` values out, remove the `-`, and change the data type to `int8` for our mergin purposes coming up in the next section.


```python
df['BGTOcc'].str.replace('-', '').astype(np.float32).isna().sum()
```




    72410




```python
%%time

df = df[df['BGTOcc'].notna()].copy()
df['BGTOcc'] = df['BGTOcc'].str.replace('-', '').astype(np.float32).astype(np.int32)
df['BGTOcc'].head()
```

    CPU times: user 1.03 s, sys: 1.7 s, total: 2.73 s
    Wall time: 2.88 s





    0    412031
    1    412031
    2    537061
    3    419041
    4    493023
    Name: BGTOcc, dtype: int32



## 2. Load Occupations Data

You can find the cleaned occupations data in our collaborative [DropBox folder here](https://www.dropbox.com/sh/k3qogf35dhxepik/AADndlDA4VZcLl5Mku0wAFRFa?dl=0).


```python
occupations_df = pd.read_csv('~/Dropbox/Burning Glass/Analysis/occupations_clean.csv',
                             dtype={'occu_code': np.int32, 'occu_text': np.str})
occupations_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>occu_code</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>110000</td>
      <td>Management Occupations</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112022</td>
      <td>Sales Managers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113021</td>
      <td>Computer and Information Systems Managers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113131</td>
      <td>Training and Development Managers</td>
    </tr>
    <tr>
      <th>4</th>
      <td>119021</td>
      <td>Construction Managers</td>
    </tr>
  </tbody>
</table>
</div>




```python
occupations_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 564 entries, 0 to 563
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   occu_code  564 non-null    int32 
     1   occu_text  564 non-null    object
    dtypes: int32(1), object(1)
    memory usage: 6.7+ KB



```python
df.shape
```




    (645891, 25)



We will merge the the main dataframe with the occupations dataframe using an inner join, and by combining `BTGOcc` with `occu_code` (which stands for occupation code).


```python
%%time

new_df = df.merge(occupations_df, 
                  left_on='BGTOcc', 
                  right_on='occu_code', 
                  how='inner')
new_df.head()
```

    CPU times: user 1.72 s, sys: 1.59 s, total: 3.31 s
    Wall time: 3.35 s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGTOcc</th>
      <th>BGTSubOcc</th>
      <th>CanonCity</th>
      <th>CanonCounty</th>
      <th>CanonEmployer</th>
      <th>CanonJobHours</th>
      <th>CanonJobTitle</th>
      <th>CanonJobType</th>
      <th>CanonSkillClusters</th>
      <th>CanonSkills</th>
      <th>CanonState</th>
      <th>CanonYearsOfExperienceLevel</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
      <th>ConsolidatedTitle</th>
      <th>DivisionCode</th>
      <th>JobDate</th>
      <th>JobID</th>
      <th>JobText</th>
      <th>LMA</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MSA</th>
      <th>MinExperience</th>
      <th>occu_code</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Sp...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-25</td>
      <td>38201188919</td>
      <td>Sales Associates\n-Foley,AL36535\n-3781 S McKe...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>38200222343</td>
      <td>AUTO SALES ASSOCIATES, ENTRY LEVEL\n\nGULF CHR...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Arlington Heights</td>
      <td>Cook</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>IL</td>
      <td>mid</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>16974.0</td>
      <td>2017-03-13</td>
      <td>38195316884</td>
      <td>Automotive Sales Associate\n\nArlington Height...</td>
      <td>DV171697|MT171698</td>
      <td>41.8792</td>
      <td>-87.9747</td>
      <td>16980: Metropolitan Statistical Area|176: Comb...</td>
      <td>3.0</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Elgin</td>
      <td>Kane</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>IL</td>
      <td>mid</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>20994.0</td>
      <td>2017-03-13</td>
      <td>38195337619</td>
      <td>Automotive Sales Associate\n\nElgin, Illinois ...</td>
      <td>DV171697|MT171698</td>
      <td>42.0363</td>
      <td>-88.2398</td>
      <td>16980: Metropolitan Statistical Area|176: Comb...</td>
      <td>3.0</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>4</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Cincinnati</td>
      <td>Hamilton</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>NaN</td>
      <td>permanent</td>
      <td>Supply Chain and Logistics: Procurement;Specia...</td>
      <td>{'Buying Experience': 'Supply Chain and Logist...</td>
      <td>OH</td>
      <td>NaN</td>
      <td>Automobile Sales &amp; Used</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Automobile Sales</td>
      <td>NaN</td>
      <td>2017-03-14</td>
      <td>38195661888</td>
      <td>Automobile Sales NEW &amp; USED\n\nJake Sweeney Ch...</td>
      <td>MT391714</td>
      <td>39.1072</td>
      <td>-84.5003</td>
      <td>17140: Metropolitan Statistical Area</td>
      <td>NaN</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
  </tbody>
</table>
</div>



Inspect the new dataframe shape.


```python
new_df.shape
```




    (540143, 27)



### Amount of Occupations per Employer on Filtered Down Data


```python
companies = new_df.groupby(['CanonEmployer'])
companies['occu_text'].agg(['count'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>CanonEmployer</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Anheuser-Busch Companies, Inc.</th>
      <td>3632</td>
    </tr>
    <tr>
      <th>Bain Company</th>
      <td>875</td>
    </tr>
    <tr>
      <th>Bank of America</th>
      <td>102030</td>
    </tr>
    <tr>
      <th>Boston Consulting Group</th>
      <td>1405</td>
    </tr>
    <tr>
      <th>Chrysler</th>
      <td>46804</td>
    </tr>
    <tr>
      <th>Citi</th>
      <td>34659</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>17455</td>
    </tr>
    <tr>
      <th>Ford Motor Company</th>
      <td>8728</td>
    </tr>
    <tr>
      <th>General Motors</th>
      <td>25734</td>
    </tr>
    <tr>
      <th>Google Inc.</th>
      <td>17425</td>
    </tr>
    <tr>
      <th>JP Morgan Chase Company</th>
      <td>134396</td>
    </tr>
    <tr>
      <th>Kraft Foods</th>
      <td>2216</td>
    </tr>
    <tr>
      <th>McKinsey &amp; Company</th>
      <td>4942</td>
    </tr>
    <tr>
      <th>Microsoft Corporation</th>
      <td>28565</td>
    </tr>
    <tr>
      <th>Morgan Stanley</th>
      <td>11973</td>
    </tr>
    <tr>
      <th>Nestle USA Incorporated</th>
      <td>19032</td>
    </tr>
    <tr>
      <th>Oliver Wyman</th>
      <td>93</td>
    </tr>
    <tr>
      <th>PepsiCo Inc.</th>
      <td>45217</td>
    </tr>
    <tr>
      <th>Roland Berger</th>
      <td>22</td>
    </tr>
    <tr>
      <th>The Goldman Sachs Group, Inc.</th>
      <td>11417</td>
    </tr>
    <tr>
      <th>Twitter</th>
      <td>3283</td>
    </tr>
    <tr>
      <th>Tyson Foods Incorporated</th>
      <td>17428</td>
    </tr>
    <tr>
      <th>Yahoo</th>
      <td>2812</td>
    </tr>
  </tbody>
</table>
</div>



Let's save the final dataset and move on to the testing all assumptions.


```python
new_df.to_csv('~/Dropbox/Burning Glass/Data/comps_occupations_merged.csv', 
              index=False)
```

## 3. Load Merged Dataset

To release some memory from your PC, make sure you restart the kernel (if on Jupyter Notebooks), load the libraries we will need, and proceed to run the code below.

If you come back to this notebook and start from this section onwards, don't forget to load all of the libraries we will use, which can be found at the beginning of this notebook.


```python
%%time

df = pd.read_csv('~/Dropbox/Burning Glass/Data/comps_occupations_merged.csv',
                 low_memory=False, parse_dates=['JobDate'], dtype=dtypes)

# Concatenate all of the datasets into one
df.reset_index(drop=True, inplace=True)
df.head(3)
```

    CPU times: user 20.9 s, sys: 13 s, total: 33.9 s
    Wall time: 44.5 s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGTOcc</th>
      <th>BGTSubOcc</th>
      <th>CanonCity</th>
      <th>CanonCounty</th>
      <th>CanonEmployer</th>
      <th>CanonJobHours</th>
      <th>CanonJobTitle</th>
      <th>CanonJobType</th>
      <th>CanonSkillClusters</th>
      <th>CanonSkills</th>
      <th>CanonState</th>
      <th>CanonYearsOfExperienceLevel</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedInferredNAICS</th>
      <th>ConsolidatedONET</th>
      <th>ConsolidatedTitle</th>
      <th>DivisionCode</th>
      <th>JobDate</th>
      <th>JobID</th>
      <th>JobText</th>
      <th>LMA</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MSA</th>
      <th>MinExperience</th>
      <th>occu_code</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Sp...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-25</td>
      <td>38201188919</td>
      <td>Sales Associates\n-Foley,AL36535\n-3781 S McKe...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Foley</td>
      <td>Baldwin</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>AL</td>
      <td>NaN</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>38200222343</td>
      <td>AUTO SALES ASSOCIATES, ENTRY LEVEL\n\nGULF CHR...</td>
      <td>MC011930</td>
      <td>30.3962</td>
      <td>-87.7019</td>
      <td>19300: Metropolitan Statistical Area</td>
      <td>NaN</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412031</td>
      <td>Retail Sales Associate (General)</td>
      <td>Arlington Heights</td>
      <td>Cook</td>
      <td>Chrysler</td>
      <td>fulltime</td>
      <td>Sales Associate</td>
      <td>permanent</td>
      <td>Sales: Specialized Sales;Specialized Skills|Co...</td>
      <td>{'Automotive Sales': 'Sales: Specialized Sales...</td>
      <td>IL</td>
      <td>mid</td>
      <td>Automotive Sales Associate</td>
      <td>336111.0</td>
      <td>41203100</td>
      <td>Sales Associate</td>
      <td>16974.0</td>
      <td>2017-03-13</td>
      <td>38195316884</td>
      <td>Automotive Sales Associate\n\nArlington Height...</td>
      <td>DV171697|MT171698</td>
      <td>41.8792</td>
      <td>-87.9747</td>
      <td>16980: Metropolitan Statistical Area|176: Comb...</td>
      <td>3.0</td>
      <td>412031</td>
      <td>Retail Salespersons</td>
    </tr>
  </tbody>
</table>
</div>



We will define a path for saving the output of all assumptions in approach 2 below.


```python
!ls ~/Dropbox
```

    [34mBurning Glass[m[m                      Icon?
    ExtendedChapter9.pdf               compressed_df.pkl
    Get Started with Dropbox Paper.url new_df.csv
    Get Started with Dropbox.pdf



```python
path_assumptions = '/Users/ramonperez/Dropbox/Burning Glass/Analysis/approach_2/'
```

Let's look at the memory usage of our new merged dataset.


```python
df.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 540143 entries, 0 to 540142
    Data columns (total 27 columns):
     #   Column                       Non-Null Count   Dtype         
    ---  ------                       --------------   -----         
     0   BGTOcc                       540143 non-null  object        
     1   BGTSubOcc                    540122 non-null  object        
     2   CanonCity                    535037 non-null  object        
     3   CanonCounty                  533867 non-null  object        
     4   CanonEmployer                540143 non-null  object        
     5   CanonJobHours                303355 non-null  object        
     6   CanonJobTitle                366347 non-null  object        
     7   CanonJobType                 304541 non-null  object        
     8   CanonSkillClusters           532116 non-null  object        
     9   CanonSkills                  540143 non-null  object        
     10  CanonState                   540126 non-null  object        
     11  CanonYearsOfExperienceLevel  349183 non-null  object        
     12  CleanJobTitle                540126 non-null  object        
     13  ConsolidatedInferredNAICS    536574 non-null  float64       
     14  ConsolidatedONET             540143 non-null  object        
     15  ConsolidatedTitle            540130 non-null  object        
     16  DivisionCode                 273912 non-null  float64       
     17  JobDate                      540143 non-null  datetime64[ns]
     18  JobID                        540143 non-null  object        
     19  JobText                      540143 non-null  object        
     20  LMA                          531863 non-null  object        
     21  Latitude                     535054 non-null  object        
     22  Longitude                    535054 non-null  object        
     23  MSA                          524880 non-null  object        
     24  MinExperience                345300 non-null  float64       
     25  occu_code                    540143 non-null  int64         
     26  occu_text                    540143 non-null  object        
    dtypes: datetime64[ns](1), float64(3), int64(1), object(22)
    memory usage: 3.4 GB


# 4. Test Assumptions

## a. For each firm, what is the number of unique occupations?

We will first take a list of the unique companies in the dataset.


```python
comps = list(df['CanonEmployer'].unique())
comps
```




    ['Chrysler',
     'Ford Motor Company',
     'JP Morgan Chase Company',
     'McKinsey & Company',
     'Boston Consulting Group',
     'Bain Company',
     'General Motors',
     'Microsoft Corporation',
     'Citi',
     'PepsiCo Inc.',
     'Tyson Foods Incorporated',
     'Nestle USA Incorporated',
     'Bank of America',
     'The Goldman Sachs Group, Inc.',
     'Morgan Stanley',
     'Kraft Foods',
     'Anheuser-Busch Companies, Inc.',
     'Google Inc.',
     'Facebook',
     'Twitter',
     'Yahoo',
     'Oliver Wyman',
     'Roland Berger']



Then, we will iterate over the list of employers, add the name of the employer and the number of unique occupations as key-value pairs to a dictionary, and then add the employer's name and the name of the unique occupations to another dictionary. In essence, we are creating two dictionaries:
- One with the employers and the amount of occupations advertised between 2016 and 2018
- Another with the employer and the name of those occupations for which they advertised roles between 2016 and 2018


```python
%%time

unique_occ_comp = {}
unique_occ_nums = {}

for comp in comps:
    unique_occs = list(df.loc[df['CanonEmployer'] == comp, 'occu_text'].unique())
    number_occs = len(df.loc[df['CanonEmployer'] == comp, 'occu_text'].unique())
    unique_occ_comp[comp] = unique_occs
    unique_occ_nums[comp] = number_occs
```

    CPU times: user 1.21 s, sys: 39.4 ms, total: 1.25 s
    Wall time: 1.3 s


Let's evaluate the results of the amount of unique occupations per firm.


```python
from pprint import pprint
pprint(unique_occ_nums)
```

    {'Anheuser-Busch Companies, Inc.': 137,
     'Bain Company': 55,
     'Bank of America': 227,
     'Boston Consulting Group': 81,
     'Chrysler': 254,
     'Citi': 197,
     'Facebook': 247,
     'Ford Motor Company': 155,
     'General Motors': 241,
     'Google Inc.': 170,
     'JP Morgan Chase Company': 242,
     'Kraft Foods': 123,
     'McKinsey & Company': 116,
     'Microsoft Corporation': 210,
     'Morgan Stanley': 134,
     'Nestle USA Incorporated': 209,
     'Oliver Wyman': 30,
     'PepsiCo Inc.': 214,
     'Roland Berger': 4,
     'The Goldman Sachs Group, Inc.': 154,
     'Twitter': 138,
     'Tyson Foods Incorporated': 206,
     'Yahoo': 90}


Let' now visualise those values by converting our dictionary into a pandas series and calling the pandas method `.plot()` on it.


```python
pd.Series(unique_occ_nums).sort_values().plot(kind='bar',
                                              rot=90, 
                                              figsize=(15, 8),
                                              fontsize=14,
                                              title='Number of Unique Occupations within a Firm between 2016-2018',
                                              grid=True,
                                              alpha=0.70);
```


![png](output_61_0.png)


Let's now look at the name of those unique occupations within a firm.


```python
pprint(unique_occ_comp['Kraft Foods'][:10])
```

    ['Retail Salespersons',
     'Cleaners of Vehicles and Equipment',
     'Automotive Service Technicians and Mechanics',
     'Sales Representatives, Wholesale and Manufacturing, Technical and Scientific '
     'Products',
     'Secretaries and Administrative Assistants, Except Legal, Medical, and '
     'Executive',
     'Light Truck or Delivery Services Drivers',
     'First-Line Supervisors of Mechanics, Installers, and Repairers',
     'Computer Programmers',
     'Accountants and Auditors',
     'Computer Occupations, All Other']


Let's save both samples to a csv file for a more in-depth evaluation.


```python
# unique occupation counts
pd.Series(unique_occ_nums).to_csv(path_assumptions + 'unique_occu_nums_per_firms.csv')

# unique occupation names
(pd.DataFrame.from_dict(unique_occ_comp, orient='index')
             .T
             .to_csv(path_assumptions + 'unique_occu_names_per_firms.csv', index=False))
```

## b. For every occupation within a company, get the unique number of job titles & names of those job titles

We will add all three columns to a variable called job_cols as it will be easier to use them later on.


```python
job_cols = ['CleanJobTitle', 'ConsolidatedTitle', 'CanonJobTitle']
```

To get the name of the jobs within each occupation at a company, we can move the variables of interest to the index and create a multi-level index. Then we select our columns above to make each job map to the companies and occupations. The next step will be to reset the index to get those columns out and repopulate the matching values to each job.


```python
# set the columns of interest in a multilevel index, this can also be thought of as
# adding dimensions to our dataset
comps_occus = df.set_index(['CanonEmployer', 'occu_text'])

# select the job title columns and bring the companies and occupations back out
comps_occus = comps_occus[job_cols].reset_index()
comps_occus.head()
```

    CPU times: user 358 ms, sys: 120 ms, total: 478 ms
    Wall time: 557 ms





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>occu_text</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedTitle</th>
      <th>CanonJobTitle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automotive Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automotive Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automotive Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automobile Sales &amp; Used</td>
      <td>Automobile Sales</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# evaluate the shape of the dataset containing duplicates
comps_occus.shape
```




    (540143, 5)



Now that we have the dataset we wanted for this assumption, we can go ahead and get rid of duplicates with the `.drop_duplicates()` method of pandas.


```python
comps_occus.drop_duplicates(inplace=True)
comps_occus.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>occu_text</th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedTitle</th>
      <th>CanonJobTitle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automotive Sales Associate</td>
      <td>Sales Associate</td>
      <td>Sales Associate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automobile Sales &amp; Used</td>
      <td>Automobile Sales</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Automobile Sales - Internet</td>
      <td>Automobile Sales/Intern</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chrysler</td>
      <td>Retail Salespersons</td>
      <td>Internet Sales/Bdc Sales/Automotive Sales</td>
      <td>Intern/Sales, Sales,Automotive Sales</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# notice the new shape of the dataset
comps_occus.shape
```




    (216517, 5)



To get the number of unique jobs we will group again our dataset by company and occupation, select the job title columns, and then apply the `.count()` method to the unique count of each.


```python
comps_occus_nums = comps_occus.groupby(['CanonEmployer', 'occu_text'])[job_cols].count()
comps_occus_nums.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CleanJobTitle</th>
      <th>ConsolidatedTitle</th>
      <th>CanonJobTitle</th>
    </tr>
    <tr>
      <th>CanonEmployer</th>
      <th>occu_text</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Anheuser-Busch Companies, Inc.</th>
      <th>Accountants and Auditors</th>
      <td>69</td>
      <td>69</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Agricultural and Food Science Technicians</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Amusement and Recreation Attendants</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Architects, Except Landscape and Naval</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Archivists</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let's save both files to our shared [Dropbox folder > Burning Glass > Analysis > approach_2 > part_b](https://www.dropbox.com/sh/faunias0nig1mya/AACc4bmAKqQHRWbvnawseCjea?dl=0).


```python
%%time

comps_occus.to_csv(path_assumptions + 'comps_occus_jobs_names.csv', index=False)
comps_occus_nums.to_csv(path_assumptions + 'comps_occus_jobs_nums.csv')
```

    CPU times: user 828 ms, sys: 130 ms, total: 959 ms
    Wall time: 1.06 s


# c. For every job title, number of occupations it appears under

Here is the column var we created a bit ago. We will use it again to get the data we need.


```python
job_cols
```




    ['CleanJobTitle', 'ConsolidatedTitle', 'CanonJobTitle']




```python
CleanJobTitle = [x for x in list(df['CleanJobTitle'].unique()) if str(x) != 'nan']
ConsolidatedTitle = [x for x in list(df['ConsolidatedTitle'].unique()) if str(x) != 'nan']
CanonJobTitle = [x for x in list(df['CanonJobTitle'].unique()) if str(x) != 'nan']
CleanJobTitle[:2], ConsolidatedTitle[:2], CanonJobTitle[:2]
```




    (['Sales Associate', 'Automotive Sales Associate'],
     ['Sales Associate', 'Automobile Sales'],
     ['Sales Associate', 'Sales Representative'])




```python
len(CleanJobTitle), len(ConsolidatedTitle), len(CanonJobTitle)
```




    (207310, 54487, 2278)



The numbers above are the amount of unique jobs per job title variable (without any missing values).

We will now create groups of all variables of interest and add the count of the amount of occupations in which a job title appears in.

### Number of Occupations in Which at Job in CanonJobTitle Appears


```python
occu_per_title_CanonJobTitle = df.groupby(['CanonEmployer', 'CanonJobTitle'])
occu_per_title_CanonJobTitle = occu_per_title_CanonJobTitle['occu_text'].count().reset_index()
occu_per_title_CanonJobTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>CanonJobTitle</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Account Manager</td>
      <td>141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Accountant</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Accounting Analyst</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Accounting Assistant</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Accounting Clerk</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>



### Number of Occupations in Which at Job in ConsolidatedTitle Appears


```python
occu_per_title_ConsolidatedTitle = df.groupby(['CanonEmployer', 'ConsolidatedTitle'])
occu_per_title_ConsolidatedTitle = occu_per_title_ConsolidatedTitle['occu_text'].count().reset_index()
occu_per_title_ConsolidatedTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>ConsolidatedTitle</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2018 Ab Inbev Gmt Scholarship</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Ab Inbev Gmt Scholarship</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Ab, Event Services</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Account Manager</td>
      <td>141</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>Accountant</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Number of Occupations in Which at Job in CleanJobTitle Appears


```python
occu_per_title_CleanJobTitle = df.groupby(['CanonEmployer', 'CleanJobTitle'])
occu_per_title_CleanJobTitle = occu_per_title_CleanJobTitle['occu_text'].count().reset_index()
occu_per_title_CleanJobTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>CleanJobTitle</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2016 Brewery Experiences Team Member</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2017 Summer Brewery Experiences Team Member</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2017 Summer Gift Shop Team Member</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2017 Summer Tour Team Member</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anheuser-Busch Companies, Inc.</td>
      <td>2018 Ab Inbev Gmt Scholarship</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's save all files as an Excel file with different sheets into our shared [Dropbox folder > Burning Glass > Analysis > approach_2 > part_c](https://www.dropbox.com/sh/gygve96olza6d31/AACoqU-5uwe56BSrQvw7h8tza?dl=0).


```python
%%time

writer = pd.ExcelWriter(path_assumptions + 'allcompanies_job_occu_nums.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
jobs_per_occu.to_excel(writer, sheet_name='jobs_per_occu', index=False)
occu_per_title_CleanJobTitle.to_excel(writer, sheet_name='CleanJobTitle', index=False)
occu_per_title_ConsolidatedTitle.to_excel(writer, sheet_name='ConsolidatedTitle', index=False)
occu_per_title_CanonJobTitle.to_excel(writer, sheet_name='CanonJobTitle', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
```

    CPU times: user 14.2 s, sys: 332 ms, total: 14.5 s
    Wall time: 14.7 s



```python
occu_per_title_CleanJobTitle[occu_per_title_CleanJobTitle['CanonEmployer'] == 'Facebook'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonEmployer</th>
      <th>CleanJobTitle</th>
      <th>occu_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66610</th>
      <td>Facebook</td>
      <td>.Net Consultant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>66611</th>
      <td>Facebook</td>
      <td>.Net Developer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>66612</th>
      <td>Facebook</td>
      <td>.Net Web Developer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>66613</th>
      <td>Facebook</td>
      <td>0005 Behavioral Health Registered Nurse</td>
      <td>1</td>
    </tr>
    <tr>
      <th>66614</th>
      <td>Facebook</td>
      <td>10-$15/Hour Summer Job 40 Hr/Week House Painti...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# d. Avg number of occupations a job title is assigned to

What we will do in this section to get the average number of occupations a job title is assigned to is to, create groups of titles and occupations, assign the n count of an occupation back into the groupped dataframe, reset the index of the groupped dataframe, and then re-group the dataset to take the mean of the n count of the different instances of occupations. A mouthfull, I know 😅, so let's look at the code instead.

## CanonJobTitle


```python
# first group the dataset by job title var and occupation
group_CanonJobTitle = df.groupby(['CanonJobTitle', 'occu_text'])

# get the ammount of occupations within that job title var using the aggregation function count
testing = group_CanonJobTitle['occu_text'].agg('count')

# reset the index twice, first to rename our column of interest and then to fully reset the df
df_CanonJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()

# examine the dataset
df_CanonJobTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CanonJobTitle</th>
      <th>occu_text</th>
      <th>occu_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.Net Application Developer</td>
      <td>Computer Programmers</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>.Net Application Developer</td>
      <td>Computer User Support Specialists</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>.Net Application Developer</td>
      <td>Web Developers</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>.Net Architect</td>
      <td>Computer Occupations, All Other</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>.Net Architect</td>
      <td>Computer Programmers</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# group the dataset again by our job title variable of interest
first_gb = df_CanonJobTitle.groupby('CanonJobTitle')

# using the occumation count variable, get the mean per job title
first_gb = first_gb['occu_count'].agg('mean')#.rename('occu_mean')

# examine the result
first_gb.head()
```




    CanonJobTitle
    .Net Application Developer    18.666667
    .Net Architect                 5.000000
    .Net Developer                57.875000
    .Net Programmer                9.250000
    .Net Team Lead                 4.000000
    Name: occu_count, dtype: float64



## CleanJobTitle


```python
# first group the dataset by job title var and occupation
group_CleanJobTitle = df.groupby(['CleanJobTitle', 'occu_text'])

# get the ammount of occupations within that job title var using the aggregation function count
testing = group_CleanJobTitle['occu_text'].agg('count')

# reset the index twice, first to rename our column of interest and then to fully reset the df
df_CleanJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()

# examine the dataset
df_CleanJobTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CleanJobTitle</th>
      <th>occu_text</th>
      <th>occu_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$22.77 Per Hour Job</td>
      <td>Cost Estimators</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&amp; - Fc Client Service Representative I</td>
      <td>Customer Service Representatives</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&amp; - Fc Client Service Representative I Teller</td>
      <td>Customer Service Representatives</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&amp; - Relationship Banker- Financial Cen</td>
      <td>New Accounts Clerks</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&amp; - Relationship Banker- Financial Center</td>
      <td>New Accounts Clerks</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# group the dataset again by our job title variable of interest
second_gb = df_CleanJobTitle.groupby('CleanJobTitle')

# using the occumation count variable, get the mean per job title
second_gb = second_gb['occu_count'].agg('mean').rename('occu_mean')

# examine the result
second_gb.head(50)
```




    CleanJobTitle
    $22.77 Per Hour Job                                                         1.0
    & - Fc Client Service Representative I                                      1.0
    & - Fc Client Service Representative I Teller                               1.0
    & - Relationship Banker- Financial Cen                                      1.0
    & - Relationship Banker- Financial Center                                   1.0
    & Bch-Relationship Banker- , Flarea - Spanish                               1.0
    & Bch-Relationship Banker- Center                                           1.0
    & Bch-Relationship Banker- Center, -Spanish                                 1.0
    & Bch-Relationship Banker-Devonaire Center                                  1.0
    & Bch-Relationship Banker-Devonaire Center, , -Spanish                      1.0
    & Brand Advertising Specialist                                              1.0
    & Kettner - Fc Client Service Representative I                              1.0
    & Kettner - Fc Client Service Representative I - Bi                         1.0
    & Kettner - Fc Client Service Representative I Teller                       1.0
    & S Fc Client Service Representative I - - , Las                            1.0
    & S Fc Client Service Representative I - Human Resources                    1.0
    & S Relationship Banker                                                     1.0
    & S Relationship Banker - - ,                                               1.0
    &L General Stores Supervisor Proc Coach                                     1.0
    &L Supervisor                                                               5.0
    &L-Production Control Coordinator                                           1.0
    &S Service Zone Manager                                                     1.0
    &S Zone Manager                                                             4.0
    's                                                                          1.0
    's Assistant Store Manager/Leader                                           1.0
    's Cashier-Host/Ess                                                         3.0
    's Center For Family Wealth Dynamics And Governance                         1.0
    's Center For Family Wealth Dynamics And Governancer Coordinator Role       1.0
    's Maintenance Technician                                                   3.0
    's Shift Leader                                                             1.0
    's Street Team                                                              1.0
    , Merchandiser ///Sign On                                                   2.0
    , Merchandiser Pt///Sign On                                                 2.0
    - Client Service Representative                                             2.0
    .Net - Back End Developer                                                   1.0
    .Net And Pega Robotics Developer                                            1.0
    .Net Application Developer                                                 15.0
    .Net Application Developer - Associate                                      3.0
    .Net Application Developer - Full Stack Developer                           1.5
    .Net Application Developer -Associate                                       1.0
    .Net Application Developer Lead                                             1.0
    .Net Application Developer, Treasury Monitoring                             2.0
    .Net Application Development Lead                                           3.0
    .Net Application Development Lead - Vice President                          1.0
    .Net Application Developmentlead                                            2.0
    .Net Application Programmer                                                 4.0
    .Net Application Programmer/Developer                                       3.0
    .Net Application Security Source Code Engineer/Developer                    5.0
    .Net Application Security Source Code Engineer/Developer - Web & Mobil      7.0
    .Net Application Security Source Code Engineer/Developer - Web & Mobile     2.0
    Name: occu_mean, dtype: float64



## ConsolidatedTitle


```python
# first group the dataset by job title var and occupation
group_ConsolidatedTitle = df.groupby(['ConsolidatedTitle', 'occu_text'])

# get the ammount of occupations within that job title var using the aggregation function count
testing = group_ConsolidatedTitle['occu_text'].agg('count')

# reset the index twice, first to rename our column of interest and then to fully reset the df
df_ConsolidatedTitle = testing.reset_index(level=1, name='occu_count').reset_index()

# examine the dataset
df_ConsolidatedTitle.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ConsolidatedTitle</th>
      <th>occu_text</th>
      <th>occu_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$22.77 Per Hour Job</td>
      <td>Cost Estimators</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'s</td>
      <td>Combined Food Preparation and Serving Workers,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>'s Center For Family Wealth Dynamics And Gover...</td>
      <td>First-Line Supervisors of Retail Sales Workers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'s Seasonal Retail Receiving, Garden State Plaza</td>
      <td>Sales Managers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'s Street Team</td>
      <td>Demonstrators and Product Promoters</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# group the dataset again by our job title variable of interest
third_gb = df_ConsolidatedTitle.groupby('ConsolidatedTitle')

# using the occumation count variable, get the mean per job title
third_gb = third_gb['occu_count'].agg('mean').rename('occu_mean')

# examine the result
third_gb.head(50)
```




    ConsolidatedTitle
    $22.77 Per Hour Job                                           1.000000
    's                                                            1.000000
    's Center For Family Wealth Dynamics And Governance           1.000000
    's Seasonal Retail Receiving, Garden State Plaza              1.000000
    's Street Team                                                1.000000
    .Net Application Developer                                   18.666667
    .Net Applicationdeveloper                                     1.000000
    .Net Applicationsdeveloper                                    2.000000
    .Net Architect                                                5.000000
    .Net C# Software Engineering Lead                             3.000000
    .Net Developer                                               57.750000
    .Net Development Lead                                         1.000000
    .Net Production Services Lead                                 1.000000
    .Net Production Support Lead                                  1.000000
    .Net Programmer                                               9.250000
    .Net Software Engineering                                     2.000000
    .Net Software Engineering Lead                                8.000000
    .Net Software Engineering Lead In | By Gigajob                1.000000
    .Net Software Engineering, Automation Testing                 1.000000
    .Net Team Lead                                                4.000000
    .Net Technical Lead                                           1.000000
    .Net Warranty Reduction - Cost Recovery Lead                  4.000000
    .Net Warranty Reduction - Cost Recovery Lead - Eng0038447     1.000000
    .Net Warranty Reduction - Cost Recovery Lead - Eng0039480     1.000000
    .Net Web Developer                                           17.000000
    .Net Wpfdeveloper                                             1.000000
    .Net/C# - Applicationsdeveloper                               1.000000
    .Net/C# -Software Engineering                                 5.000000
    .Net/C# Applicationsdeveloper                                 1.000000
    .Net/C# Software Engineering                                  2.000000
    .Net/C# Software Engineering Lead                             2.000000
    .Net/C#- Software Engineering                                 1.000000
    .Net/Netoxygendeveloper                                       1.000000
    .Net/Sharepointdeveloper                                      1.000000
    .Net/Sql Applicationsdeveloper                                1.000000
    .Net/Work Event                                               1.000000
    009100 Direct Support Professional - Prn                      1.000000
    10-$15/Hour Summer Job 40 Hr/Week House Painting Co           1.000000
    110536555                                                     1.000000
    1469770459                                                    1.000000
    15008230                                                      1.000000
    15008239                                                      1.000000
    15008325                                                      1.000000
    15008747                                                      1.000000
    15008760                                                      1.000000
    15008906                                                      1.000000
    15008959                                                      1.000000
    150118745                                                     1.000000
    150124347                                                     1.000000
    16000286                                                      1.000000
    Name: occu_mean, dtype: float64



Now that we have the data we needed, let's proceed to save it to our collaborative [Dropbox folder > Burning Glass > Analysis > approach_2 > part_d](https://www.dropbox.com/sh/iy1of1f2vukwjgu/AABBaCZSugocVGhjol5Lzl85a?dl=0).


```python
%%time

writer = pd.ExcelWriter(path_assumptions + 'average_occu_jobs.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_CanonJobTitle.to_excel(writer, sheet_name='CanonJobTitle', index=False)
first_gb.to_excel(writer, sheet_name='Avg_CanonJobTitle')

df_CleanJobTitle.to_excel(writer, sheet_name='CleanJobTitle', index=False)
second_gb.to_excel(writer, sheet_name='Avg_CleanJobTitle')

df_ConsolidatedTitle.to_excel(writer, sheet_name='ConsolidatedTitle', index=False)
third_gb.to_excel(writer, sheet_name='Avg_ConsolidatedTitle')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
```

    CPU times: user 23.7 s, sys: 1.33 s, total: 25.1 s
    Wall time: 26.5 s


## **Additions from meeting on Friday, June 19**
1. List of job titles (consolidated & clean) e.g., for Facebook, which appear less than median (I think it was 2) times. [Link to answer](https://www.dropbox.com/s/ekqjjb4t00l8b74/fb_job_var_nums.xlsx?dl=0)
2. For a sample firm, xls with list of unique occupations (we need to read those and see whether they  make sense in terms of functions / divisions)
[Link to answer](https://www.dropbox.com/s/ccll89c1vhqf6av/sample_firm_unique_occu.csv?dl=0)
3. Avg number of occupations a job title (both consolidated & clean) is assigned to (for a sample firm and across firms in the sample) [Link to answer](https://www.dropbox.com/sh/w2j41ka4hsgx0gf/AAAdEGOsjI9_gK8__ZeVHE5Ta?dl=0)
    - For each job title in a company, how many different occupations this job map into? **This can be found by sorting the file with the data**
    - For each of the 20 most common canon job titles, extract all unique clean job titles that correspond to them **This can be found by sorting the file with the data**
4. Xls with examples of those job titles assigned to multiple occupations: is there an occupation they are primarily associated to, and the others are just some noise. **The answer to this question can be found in the same file as that of point 3 of this section.**

## Facebook CanonJobTitle


```python
group_CanonJobTitle = df[df['CanonEmployer'] == 'Facebook'].groupby(['CanonJobTitle', 'occu_text'])
testing = group_CanonJobTitle['occu_text'].agg('count')
df_CanonJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
first_gb = df_CanonJobTitle.groupby('CanonJobTitle')
first_gb = first_gb['occu_count'].agg('mean').rename('occu_mean')
first_gb.head()
```




    CanonJobTitle
    .Net Developer        4.0
    .Net Web Developer    1.0
    3D Generalist         4.0
    ASP .Net Developer    1.0
    Account Director      4.0
    Name: occu_mean, dtype: float64



## CleanJobTitle


```python
group_CleanJobTitle = df[df['CanonEmployer'] == 'Facebook'].groupby(['CleanJobTitle', 'occu_text'])
testing = group_CleanJobTitle['occu_text'].agg('count')
df_CleanJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
second_gb = df_CleanJobTitle.groupby('CleanJobTitle')
second_gb = second_gb['occu_count'].agg('mean').rename('occu_mean')
second_gb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CleanJobTitle</th>
      <th>occu_text</th>
      <th>occu_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.Net Consultant</td>
      <td>Computer Systems Analysts</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>.Net Developer</td>
      <td>Computer Programmers</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>.Net Web Developer</td>
      <td>Web Developers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0005 Behavioral Health Registered Nurse</td>
      <td>Registered Nurses</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10-$15/Hour Summer Job 40 Hr/Week House Painti...</td>
      <td>Vocational Education Teachers, Postsecondary</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## ConsolidatedTitle


```python
group_ConsolidatedTitle = df[df['CanonEmployer'] == 'Facebook'].groupby(['ConsolidatedTitle', 'occu_text'])
testing = group_ConsolidatedTitle['occu_text'].agg('count')
df_ConsolidatedTitle = testing.reset_index(level=1, name='occu_count').reset_index()
third_gb = df_ConsolidatedTitle.groupby('ConsolidatedTitle')
third_gb = third_gb['occu_count'].agg('mean').rename('occu_mean')
third_gb.head()
```




    ConsolidatedTitle
    .Net Developer                                         4.0
    .Net Web Developer                                     1.0
    10-$15/Hour Summer Job 40 Hr/Week House Painting Co    1.0
    1469770459                                             1.0
    1776076392                                             1.0
    Name: occu_mean, dtype: float64




```python
%%time

writer = pd.ExcelWriter(path_assumptions + 'fb_average_occu_jobs.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_CanonJobTitle.to_excel(writer, sheet_name='FB_CanonJobTitle', index=False)
first_gb.to_excel(writer, sheet_name='FB_Avg_CanonJobTitle')

df_CleanJobTitle.to_excel(writer, sheet_name='FB_CleanJobTitle', index=False)
second_gb.to_excel(writer, sheet_name='FB_Avg_CleanJobTitle')

df_ConsolidatedTitle.to_excel(writer, sheet_name='FB_ConsolidatedTitle', index=False)
third_gb.to_excel(writer, sheet_name='FB_Avg_ConsolidatedTitle')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
```

    CPU times: user 1.25 s, sys: 317 ms, total: 1.57 s
    Wall time: 1.59 s



```python

```

## McKinsey CanonJobTitle


```python
df['CanonEmployer'].unique()
```




    array(['Chrysler', 'Ford Motor Company', 'JP Morgan Chase Company',
           'McKinsey & Company', 'Boston Consulting Group', 'Bain Company',
           'General Motors', 'Microsoft Corporation', 'Citi', 'PepsiCo Inc.',
           'Tyson Foods Incorporated', 'Nestle USA Incorporated',
           'Bank of America', 'The Goldman Sachs Group, Inc.',
           'Morgan Stanley', 'Kraft Foods', 'Anheuser-Busch Companies, Inc.',
           'Google Inc.', 'Facebook', 'Twitter', 'Yahoo', 'Oliver Wyman',
           'Roland Berger'], dtype=object)




```python
group_CanonJobTitle = df[df['CanonEmployer'] == 'McKinsey & Company'].groupby(['CanonJobTitle', 'occu_text'])
testing = group_CanonJobTitle['occu_text'].agg('count')
df_CanonJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
first_gb = df_CanonJobTitle.groupby('CanonJobTitle')
first_gb = first_gb['occu_count'].agg('mean').rename('occu_mean')
first_gb.head()
```




    CanonJobTitle
    Account Director              4.0
    Account Manager              18.0
    Accountant                    8.5
    Accounts Receivable Clerk     1.0
    Actuarial Consultant         25.0
    Name: occu_mean, dtype: float64



## CleanJobTitle


```python
group_CleanJobTitle = df[df['CanonEmployer'] == 'McKinsey & Company'].groupby(['CleanJobTitle', 'occu_text'])
testing = group_CleanJobTitle['occu_text'].agg('count')
df_CleanJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
second_gb = df_CleanJobTitle.groupby('CleanJobTitle')
second_gb = second_gb['occu_count'].agg('mean').rename('occu_mean')
second_gb.head()
```




    CleanJobTitle
    Account Manager                              2.0
    Account Manager, Cpg                         1.0
    Account Satisfaction Director                4.0
    Account Strategist                           2.0
    Account Strategist Leading Digital Agency    1.0
    Name: occu_mean, dtype: float64



## ConsolidatedTitle


```python
group_ConsolidatedTitle = df[df['CanonEmployer'] == 'McKinsey & Company'].groupby(['ConsolidatedTitle', 'occu_text'])
testing = group_ConsolidatedTitle['occu_text'].agg('count')
df_ConsolidatedTitle = testing.reset_index(level=1, name='occu_count').reset_index()
third_gb = df_ConsolidatedTitle.groupby('ConsolidatedTitle')
third_gb = third_gb['occu_count'].agg('mean').rename('occu_mean')
third_gb.head()
```




    ConsolidatedTitle
    Account Director                4.0
    Account Manager                18.0
    Account Strategist              2.0
    Account Strategist, Digital     1.0
    Accountant                      8.5
    Name: occu_mean, dtype: float64




```python
%%time

writer = pd.ExcelWriter(path_assumptions + 'mckinsey_average_occu_jobs.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_CanonJobTitle.to_excel(writer, sheet_name='MK_CanonJob', index=False)
first_gb.to_excel(writer, sheet_name='MK_Avg_CanonJob')

df_CleanJobTitle.to_excel(writer, sheet_name='MK_CleanJob', index=False)
second_gb.to_excel(writer, sheet_name='MK_Avg_CleanJob')

df_ConsolidatedTitle.to_excel(writer, sheet_name='MK_Consolid', index=False)
third_gb.to_excel(writer, sheet_name='MK_Avg_Consolid')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
```

    CPU times: user 247 ms, sys: 13.7 ms, total: 260 ms
    Wall time: 261 ms



```python

```

## Microsoft CanonJobTitle


```python
group_CanonJobTitle = df[df['CanonEmployer'] == 'Microsoft Corporation'].groupby(['CanonJobTitle', 'occu_text'])
testing = group_CanonJobTitle['occu_text'].agg('count')
df_CanonJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
first_gb = df_CanonJobTitle.groupby('CanonJobTitle')
first_gb = first_gb['occu_count'].agg('mean').rename('occu_mean')
first_gb.head()
```




    CanonJobTitle
    .Net Developer         7.666667
    3D Generalist          3.000000
    Account Director       3.333333
    Account Executive    330.000000
    Account Manager      140.750000
    Name: occu_mean, dtype: float64



## CleanJobTitle


```python
group_CleanJobTitle = df[df['CanonEmployer'] == 'Microsoft Corporation'].groupby(['CleanJobTitle', 'occu_text'])
testing = group_CleanJobTitle['occu_text'].agg('count')
df_CleanJobTitle = testing.reset_index(level=1, name='occu_count').reset_index()
second_gb = df_CleanJobTitle.groupby('CleanJobTitle')
second_gb = second_gb['occu_count'].agg('mean').rename('occu_mean')
second_gb.head()
```




    CleanJobTitle
    .Net Associate Consultant -Federal Space    1.0
    .Net Consultant                             1.0
    .Net Consultant - Federal Space             1.0
    .Net Developer                              8.0
    .Net Developer - Azure                      1.0
    Name: occu_mean, dtype: float64



## ConsolidatedTitle


```python
group_ConsolidatedTitle = df[df['CanonEmployer'] == 'Microsoft Corporation'].groupby(['ConsolidatedTitle', 'occu_text'])
testing = group_ConsolidatedTitle['occu_text'].agg('count')
df_ConsolidatedTitle = testing.reset_index(level=1, name='occu_count').reset_index()
third_gb = df_ConsolidatedTitle.groupby('ConsolidatedTitle')
third_gb = third_gb['occu_count'].agg('mean').rename('occu_mean')
third_gb.head()
```




    ConsolidatedTitle
    .Net Developer                                     7.666667
    1St Party Manger, World-Wide Enterprise Devices    4.000000
    2019 Microsoft Ai Residency Program                3.000000
    365Dynamics Crm Business Analysis                  1.000000
    3D Generalist                                      3.000000
    Name: occu_mean, dtype: float64




```python
%%time

writer = pd.ExcelWriter(path_assumptions + 'msft_average_occu_jobs.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_CanonJobTitle.to_excel(writer, sheet_name='MSFT_CanonJob', index=False)
first_gb.to_excel(writer, sheet_name='MSFT_Avg_CanonJob')

df_CleanJobTitle.to_excel(writer, sheet_name='MSFT_CleanJob', index=False)
second_gb.to_excel(writer, sheet_name='MSFT_Avg_CleanJob')

df_ConsolidatedTitle.to_excel(writer, sheet_name='MSFT_Consolid', index=False)
third_gb.to_excel(writer, sheet_name='MSFT_Avg_Consolid')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
```

    CPU times: user 1.58 s, sys: 238 ms, total: 1.82 s
    Wall time: 1.83 s



```python

```
