# Approach 2: O*NET occupations

Idea: Get a list of O*NET occupations and their associated job titles (there are over 800 occupations and at least 5-10 titles for each). Get humans to assign each job title within and then across occupations to its likely hierarchical level in a firm, regardless of which firm it’s in. Use continuous bag of words model to expand the job titles in O*NET to the full job title list in burning glass. Once done, we can simply assign levels to each job title in BG based on coder opinion. Then we count the number of unique levels in the firm and consider firms with a larger number of unique levels as ‘more hierarchical’.

**[Next Section](https://ramonprz01.github.io/bg-insead-project/analysis/approach_2/code/)**

**Assumptions to test:**

Distribution of job titles within occupations: how many job titles correspond to each O*NET code from the BG data (occupation == O*NET code)  
1. For each firm, the number of occupations (unique) (BGTOcc)
2. For every company-BGTOcc, unique number of job titles & list of those job titles. I reinterpreted this one as “For every company and every BGTOcc within it, get unique number of job titles & list of those job titles”
3. For every job title, number of occupations it appears under
4. Not just the # of jobs and # occupations but ideally the # of job titles matched to an occupation. How many unique job titles do you see in an occupation?

**Additions from meeting on Friday, June 19:**
1. List of job titles (consolidated & clean) e.g., for Facebook, which appear less than median (I think it was 2) times 
Link to answer
2. For a sample firm, xls with list of unique occupations (we need to read those and see whether they  make sense in terms of functions / divisions)
Link to answer
3. Avg number of occupations a job title (both consolidated & clean) is assigned to (for a sample firm and across firms in the sample)
    - For each job title in a company, how many different occupations this job map into?
    - For each of the 20 most common canon job titles, extract all unique clean job titles that correspond to them
4. Xls with examples of those job titles assigned to multiple occupations: is there an occupation they are primarily associated to, and the others are just some noise


**Steps:**

- From O*NET, extract all reported job titles
- Classify O*NET job titles into a hierarchy--to be done by:
    - RA, with random validation of subsample [could be a good starting point to see whether the method works]
    - AMTurkers / INSEAD Career Services / INSEAD Alumni
- Use continuous bag-of-word approach to map BG job titles into O*NET job titles (based on synonyms), and therefore into the related hierarchical level 

**Open questions:**
1. How time consuming? How expensive? 
2. Would a bullet-proof mapping of O*NET job titles into hierarchical levels require an expensive AMTurk task? 
3. One issue with the above is that there will be multiple job titles in the same level - e.g. lawyers and finance professionals and marketing managers may all end up getting a rank of 7. How comfortable are we to assume this is likely true for all firms? If it’s mostly noise - that’s probably fine, but it may not be just noise. 
4. What do job titles miss about hierarchical structure? E.g. will we end up having firms with only levels 3 7 and 11? While others have 1 2 3 5 7 8 9 11? And how do we think about this? Can we classify the latter as being more hierarchical?
5. When MTurkers rate job titles, will they conflate expertise-based and authority-based hierarchies? This may be a significant problem if we use this as a stand-alone solution.
6. About O*NET: The biggest issue with this approach is that it’s not firm-specific; so if lawyers get a ranking of 7 (i.e. people perceive them to be 6 levels above interns, for instance), that is the ranking of lawyer in all firms, but clearly we want to look across firms.
