# Burning Glass Project

This repo contains all of the work for the burning glass research project at INSEAD. The following table of contents will guide you to the different sections of this project, which are mainly subdivided into data gathering, cleaning, and analysis.

Explanations and examples on how to run the code can be found within each section.

## Table of Contents

1. Objective
2. Data
    - [Cleaning](https://github.com/ramonprz01/bg-insead-project/cleaning)
    - Samples
3. [Analysis](https://github.com/ramonprz01/bg-insead-project/tree/master/analysis)
    - [Approach 1: Keyword search in job add text](https://github.com/ramonprz01/bg-insead-project/analysis/approach_1)
    - [Approach 2: O*NET occupations](https://github.com/ramonprz01/bg-insead-project/analysis/approach_2)
    - [Approach 3: Occupations]()
    - Approach 4: Network Analysis
    - Approach 5: Developing Prototypical Jobs - Similarity of job ads and O*NET descriptions
    - Approach 6: Job titles & Matching into hierarchical levels
4. Analysis at Scale
    - Dask
    - AWS
5. Advanced Analysis
    - Machine Learning
        - Supervised Learning
        - Unsupervised Learning
    - Deep Learning
        - Supervised Learning
        - Unsupervised Learning
6. Results
7. Further Ideas


## 1. Objectives
{:toc}

> Infer organizational structure (vertical and horizontal) from BurningGlass data

What are we really after? It seems our goal is to measure two things:
How many layers of reporting are there in a given firm? [VERTICAL]
How many divisions are there in a given firm? [HORIZONTAL]

Or, can we capture variance across companies in terms of:
Hierarchical layers
Breadth of divisions

There are many ways to try to get at these constructs, but none are perfect. We describe here five options, along with their assumptions and drawbacks.

However, we don’t think that we should pick one just yet because we actually lack enough information about whether the assumptions for each will be satisfied. So instead, we propose to test the assumptions of the first 3 approaches listed below on two different data extracts:
A semi-random selection of firms with a complete history of all their ads [this is to check how much within-firm variation we can get on various measures]
A random (or complete, if computationally quick) sample of all ads in a given time-period, e.g. one year of data for all firms [this is to check variation across firms]

We list the assumptions we aim to test for the first 3 approaches as we describe each below. 

We are leaving approaches 4 and 5 out for now because they will depend on the checks we do for the first three, and are both much more complicated.

## 2. Data
{:toc}

The dataset was purchased from Burning Glass Technologies and it is a compilation of large number of job adds that have been in circulation for the last 11 years (2010-2020) and 2007. The full, uncompressed size of the dataset is about 1TB and the full shape is a bit over a quarter of a billion rows and 57 variables.

Click on the link above to find more information about the variables.

## 3. [Analysis](https://github.com/ramonprz01/bg-insead-project/analysis)
{:toc}

## 4. Analysis at Scale
{:toc}

## 5. Advanced Analysis
{:toc}

## 6. Results
{:toc}

## 7. Further Ideas
{:toc}