# Burning Glass Project

This repo contains all of the work for the burning glass research project at INSEAD. The following table of contents will guide you to the different sections of this project, which are mainly subdivided into data gathering, cleaning, and analysis.

Explanations and examples on how to run the code can be found within each section.

## Table of Contents

1. Objective
2. Data
    - Cleaning
    - Samples
3. Approaches
    - Keyword Search in Job Add Text
    - O*NET Occupations
    - Occupations
    - Network
    - Job titles & Matching into hierarchical levels
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
