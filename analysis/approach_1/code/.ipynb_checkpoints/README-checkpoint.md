# Approach 1

**Idea:** We search for keywords in the ads related to hierarchy and separate these into ‘upward looking’ and ‘downward looking’ directions. We classify jobs as ‘high’ in the hierarchy if they contain only downward-looking keywords; we classify jobs as ‘low’ in the hierarchy if they only contain upward-looking keywords; we classify those with at least one instance of each keyword as ‘middle’.

[View or Download Notebook in NBViewer](https://nbviewer.jupyter.org/github/ramonprz01/bg-insead-project/blob/master/analysis/approach_1/code/analysis_approach1.ipynb)


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import re
from typing import Union, List
import nltk
import concurrent.futures

pd.set_option('display.max_columns', None) 

%matplotlib inline
```

For this approach we don't need all of the variables so we will use the ones below for simplicity and speed.


```python
small_list = ['JobID', 'CleanJobTitle', 'JobText', 'CanonEmployer', 'JobDate']
```

Assigning the data types before loading them helps us deal with the complexity found in some of the sample files, so we will be passing a dictionary with the column and its respective data type to the `.read_csv()` method.


```python
dtypes = {'JobID': np.str, 'CleanJobTitle': np.str, 'JobDomain': np.str, 
          'CanonCity': np.str, 'CanonCountry': np.str, 'CanonState': np.str, 
          'JobText': np.str, 'JobURL': np.str, 'PostingHTML': np.float64, 
          'Source': np.str, 'JobReferenceID': np.str, 'Email': np.str, 
          'CanonEmployer': np.str, 'Latitude': np.str, 'Longitude': np.str, 
          'CanonIntermediary': np.str, 'Telephone': np.str, 'CanonJobTitle': 'object', 
          'CanonCounty': np.str, 'DivisionCode': np.float64, 'MSA': np.str, 'LMA': np.str,
          'InternshipFlag': np.str, 'ConsolidatedONET': np.float64, 'CanonCertification': np.str, 
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

Add below the path to the dataset and the dataset for the week you'd like to test.


```python
path = '/Volumes/LaCie SSD/bgdata/data_18/'
dataset = 'data_18_0806_0812.csv'
```


```python
%%time

df = pd.read_csv(path + dataset, # combine the two above
                 low_memory=False, # since these are large files we need more memory
                 parse_dates=['JobDate'], # parse the dates for simplicity
                 usecols=small_list, # use our small list of vars
                 dtype=dtypes # assign the data types
                ) 
```

If you would like to check the true memory the dataset is occupying in your computer, use the following line of code. You will also get information regarding missing values, shape of the dataframe, and data types.


```python
df.info(memory_usage='deep')
```

To get a different perspective of the missing values, calculate the percentage of missing values for each column.


```python
df.isnull().sum() / df.shape[0] * 100
```

Filter out observations without a job description since we are not interested in those.


```python
df = df[df['JobText'].notna()].copy()
```

Clean the `JobText` column, compute the length (by characters) of the job descriptions, and convert the `clean_text` column to lower case.


```python
%%time

df['clean_text'] = df['JobText'].apply(lambda x: ' '.join(list(filter(None, x.split('\n')))))
df['len_text'] = df['clean_text'].apply(len)
df['low_clean'] = df['clean_text'].apply(lambda x: x.lower())
```

### Keywords

Here are the keywords for our first approach in two separate lists, one for downward looking and for upward looking words.


```python
down_ward = [' will supervise ', 'supervises', ' interns ', ' intern ',
             ' guides ', ' mentors ', ' leads ', ' lead ', 'oversees', 
             'will guide', ' be in charge of ', ' mentor ', 'coaching',
             'mentoring', 'coordinating', 'building teams', 'guiding',
             'advising', 'setting performance standards', 'resolving conflicts',
             'responsibility for outcomes', 'directs', 'appoints', 'instructs',
             'recruits', 'manages'
]

up_ward = [' interns ', ' intern ', 'reports to ', 'report to ', 'answers to', 
           ' managed by ', ' responds to ', ' directed by ', ' receives guidance ', 
           ' supervised by ', 'assists', 'supports', 'helps']
```

##### Keywords approach.

1. Identify the keywords above
2. Convert the boolean result into integer type
3. Replace downward looking keywords with a 3
4. Subtract upward looking from downward looking to get the difference (mid-level keyword match)
5. Replace negative instances of upward looking with a positive 1
6. Change 0's to `NaN`'s
7. Create a bucket with labels
    - High == downward looking
    - mid == mid
    - low == upward looking
8. Print value counts


```python
%%time

# the two lines below check for first instance of a keyword OR the next OR the next ...
df['down_ward'] = df['low_clean'].str.contains(' will supervise | supervises | interns | intern | guides | mentors | leads | lead | oversees | will guide | be in charge of | mentor | coaching | mentoring | coordinating | building teams | guiding | advising | setting performance standards | resolving conflicts | responsibility for outcomes | directs | appoints | instructs | recruits | manages', regex=True)
df['upward'] = df['low_clean'].str.contains(' interns | intern | reports to | report to | answers to | managed by | responds to | directed by | receives guidance | supervised by | assists | supports | helps', regex=True)

# change type
df['upward'] = df['upward'].astype(np.int8)
df['down_ward'] = df['down_ward'].astype(np.int8)

# replace nums
df['down_ward'] = df['down_ward'].replace(1, 3)

# create mid
df['all_levels'] = (df['down_ward'] - df['upward'])

# replace negative values and 0s
df['all_levels'] = df['all_levels'].replace(-1, 1)
df['all_levels'] = df['all_levels'].replace(0, np.nan)

# label the buckets
labels_dict = {1.0: 'low', 2.0: 'mid', 3.0: 'high'}
df['bucket_label'] = df['all_levels'].map(labels_dict)

df['bucket_label'].value_counts()
```

## Smaller sample for testing assumption

Fiter out of the dataset the missing values from our `bucket_label` var.


```python
%%time

df_dos = df[df['bucket_label'].notna()].copy()
```

Release some memory from your computer by deleting the df var.


```python
del df
```

Take a random sample of 50k. You can adjust this to your needs below.


```python
df_dos = df_dos.sample(50000)
df_dos.shape
```


```python
# stop_words = nltk.corpus.stopwords.words('english')
```

Deep cleaning function. Notice that we want to keep the stopwords in so that is commented out.


```python
def normalize_doc(doc):
    """
    This function normalizes your list of documents by taking only
    words, numbers, and spaces in between them. It then filters out
    stop words if you want to.
    """
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens]
    # filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

corp_normalizer = np.vectorize(normalize_doc)
```

Clean the clean text using all of the cores in your computer. Notice that we pass in the array as a numpy array for speed.


```python
%%time

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(corp_normalizer, df_dos['clean_text'].values)
```

Extract the elements out and assign them back to the same variable. `.map()` is a lazy evaluator so until we call what we need it won't give us anything in return.


```python
%%time

extract_results = [text for text in results]
df_dos['low_clean'] = extract_results
```

Confirm results by checking out the first add.


```python
extract_results[0]
```

Since the above is a numpy array inside a list, let's change that in the dataframe to a string and reasign it to the same variable.


```python
df_dos['low_clean'] = df_dos['low_clean'].astype(np.str)
```

Check out the lenght of a job description in character terms.


```python
df_dos['len_text'].describe()
```

Get every word instance as a boolean variable and add it back as a column to the dataframe.


```python
%%time

# Here we iterate throught the list of words
for word in down_ward: # and assign the keyword as a variable and a 1 if the word was found
    df_dos[word.strip()] = df_dos['low_clean'].str.contains(word) # 0 if not
    
# Here we iterate throught the list of words
for word in up_ward: # and assign the keyword as a variable and a 1 if the word was found
    df_dos[word.strip()] = df_dos['low_clean'].str.contains(word) # 0 if not
```

Create lists of the keywords without spaces in them.


```python
up_stripped = [w.strip() for w in up_ward]
down_stripped = [w.strip() for w in down_ward]
```

Sum up the amount of keyword appearances in an job description/observation.


```python
df_dos['up_instances'] = df_dos.loc[:, up_stripped].sum(axis=1)
df_dos['up_instances'].head()
```


```python
df_dos['down_instances'] = df_dos.loc[:, down_stripped].sum(axis=1)
df_dos['down_instances'].head()
```


```python
df_dos.head()
```

Get the first 60 characters of the instance where the keywords appeared. If you would like to see a bigger portion of the text, update the parameter below.

You can run this in two ways and the uncommented one is the fastest one.


```python
# %%time

# def get_words(word: str, string: str) -> Union[str, None]:
#     if word in string:
#         return string[string.index(word):string.index(word)+60]
    
# for word in up_ward:
#     df_dos[word.strip()] = df_dos['low_clean'].apply(lambda x: get_words(word, x))
    
# for word in down_ward:
#     df_dos[word.strip()] = df_dos['low_clean'].apply(lambda x: get_words(word, x))
```


```python
def get_words(word: str, string: str) -> Union[str, None]:
    if word in string:
        return string[string.index(word):string.index(word) + 60]

def get_some_text(list_of_words: List[str], data: pd.DataFrame, column: str) -> pd.DataFrame:
    for word in list_of_words:
        data[word.strip()] = data[column].apply(lambda x: get_words(word, x))
    return data
```


```python
%%time

df_dos = get_some_text(down_stripped, df_dos, 'low_clean')
df_dos = get_some_text(up_stripped, df_dos, 'low_clean')
```


```python
df_dos.head()
```

Check out the memory of your dataset and save it with your desired name.


```python
df_dos.info(memory_usage='deep')
```


```python
new_name = 'new_data.csv'
```


```python
%%time

df_dos.to_csv(path + new_name, index=False)
```
