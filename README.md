# Burning Glass Project

This repo contains all of the work for the burning glass research project at INSEAD. The following table of contents will guide you to the different sections of this project, which are mainly subdivided into data gathering, cleaning, and analysis.

Explanations and examples on how to run the code can be found within each section.

## Table of Contents

1. Objective
2. [Data](https://ramonprz01.github.io/bg-insead-project/data)
3. [Cleaning](https://ramonprz01.github.io/bg-insead-project/cleaning)
4. [Analysis](https://ramonprz01.github.io/bg-insead-project/analysis)


## 1. Objectives

> Infer organizational structure (vertical and horizontal) from job descriptions

What are we really after? It seems our goal is to measure two things:
How many layers of reporting are there in a given firm? [VERTICAL]
How many divisions are there in a given firm? [HORIZONTAL]

Or, can we capture variance across companies in terms of:
- Hierarchical layers
- Breadth of divisions

There are many ways to try to get at these constructs, but none are perfect.

## 2. [Data](https://ramonprz01.github.io/bg-insead-project/data)

The dataset was purchased from Burning Glass Technologies and it is a compilation of large number of job ads that have been posted by many companies over the last 13 years (2007-2020) and 2007.

The data comes with 57 variables and the dictionary is available in the data folder above.

### Size
- ~1TB for the full uncompressed data. (CSV files)
- ~650GB for the cleaned datasets. More details on the cleaning are available in the cleaning folder above. (CSV files)
- ~150GB for the cleaned datasets composed of only the observations with salary info in them. (parquet files)
- ~55GB for the cleaned 76k companies datasets. (parquet files)

Click on the link above to find more information about the variables.

## 3. [Cleaning](https://ramonprz01.github.io/bg-insead-project/cleaning)


## 4. [Analysis](https://ramonprz01.github.io/bg-insead-project/analysis)

### Here is where to find what?

1. Clusters based on k-means 
    - Inside analysis/approach_8 folder
    - Notebooks 09 and 09p2
2. Occurrence of up-/downward-directed terms in ads
    - Inside analysis/approach_8 folder
    - Notebooks 07 and 
Occurrences of ‘reports to’ in ads
Number of salary brackets / peaks in the distribution
Managerial intensity in job ads at the firm-year level
Number of unique job posts, occupations and job titles per firm-week
Any descriptives that you’ve produced at the industry level (e.g. 2- and 4-digit NAICS codes)
How many companies hire in each code each year
How many unique job posts there are
Any details on missing data for these