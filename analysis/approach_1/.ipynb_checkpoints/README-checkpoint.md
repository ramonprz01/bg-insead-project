# Approach 1 Rundown

![approach1](https://media.giphy.com/media/3ov9kacqGycKQRH6Vy/giphy.gif)

#### [Go Straight to the Code](https://ramonprz01.github.io/bg-insead-project/analysis/approach_1/code)

## In a Nutshell - > Keyword Search in Job Add Text

**Idea:** We search for keywords in the ads related to hierarchy and separate these into ‘upward looking’ and ‘downward looking’ directions. We classify jobs as ‘high’ in the hierarchy if they contain only downward-looking keywords; we classify jobs as ‘low’ in the hierarchy if they only contain upward-looking keywords; we classify those with at least one instance of each keyword as ‘middle’.

We construct the dictionary of keywords ourselves, with the help of O*NET (but not in a systematic way for now, just using our judgment). However, if this approach works, we can expand the dictionary in a systematic and defensible way based on prior research (e.g. Atalay et al., 2019).


### Assumptions to test (in order of priority): 
1. Being able to separate homonyms (especially nouns from verbs)
2. Being able to separate working with people internal to the organization from external ones
3. Number of postings matching keywords as % of total--we can go with the current list of words (the most conservative one). Define it 
    - In sample of 25 companies:
        - ~50% of postings contain the words we defined
        - 241521 Contains downward looking words
        - 100,400 Both downward and upward
        - 61000 Contain upward looking words
    - In 2018 sample:
        - ~60% of postings contain the words we defined

4. Variance on layers captured (high / middle / low) within and across the companies: the higher the better

#### Answers for the assumptions above can be found in two spreadsheets in the collaborative dropbox folder. [Click here to go to that folder.](https://www.dropbox.com/sh/vwee0j4ifv38nzn/AABfrKU_WB7azSvOud_iryiOa?dl=0)


### Step 1: Generate initial dictionaries:
1. For up-ward looking (i.e., employee reporting to someone else) and 
2. Down-ward looking (i.e., employee managing someone else) to search in BG job add text

Current dictionary (and suggested mapping into up-ward and down-ward looking)

### Step 2: Expand the dictionaries

**Possible solutions [All will require support from RA, A and B would be more intensive]**

a. Using O*NET

Go through relevant jobs in each of the categories below & select verbs that refer to hierarchical interactions. 

Work activities → Interacting with others (https://www.onetonline.org/find/descriptor/browse/Work_Activities/4.A.4/)

**One could get at the relevant jobs from the skill pathway (Skills → Management of Personnel)**

Interesting categories to look at: 

- Coaching and Developing Others — Identifying the developmental needs of others and coaching, mentoring, or otherwise helping others to improve their knowledge or skills.
- Communicating with Supervisors, Peers, or Subordinates — Providing information to supervisors, co-workers, and subordinates by telephone, in written form, e-mail, or in person.
- Coordinating the Work and Activities of Others — Getting members of a group to work together to accomplish tasks.
- Developing and Building Teams — Encouraging and building mutual trust, respect, and cooperation among team members.
- Guiding, Directing, and Motivating Subordinates — Providing guidance and direction to subordinates, including setting performance standards and monitoring performance.
- Monitoring and Controlling Resources — Monitoring and controlling resources and overseeing the spending of money.
- Resolving Conflicts and Negotiating with Others — Handling complaints, settling disputes, and resolving grievances and conflicts, or otherwise negotiating with others.
- Staffing Organizational Units — Recruiting, interviewing, selecting, hiring, and promoting employees in an organization.
- Training and Teaching Others — Identifying the educational needs of others, developing formal educational or training programs or classes, and teaching or instructing others.

b. **Asking RA to search synonyms from thesaurus of words in table above**


c. **Crowdsourcing it (AMTurk) -->** Asking AMTurkers to generate words that describe the actions of a manager, and the actions of a subordinate

### Step 3: Classify jobs in BG based on the dictionary
1. Use the following rules:   
    - If job description matches up-ward looking dictionary ONLY: level is LOW
    - If job description matches BOTH up-ward & down-ward looking dictionaries: level is MIDDLE
    - If job description matches down-ward looking dictionary ONLY: level is HIGH

2. Within vs. across firms:  
    - If, within firm, we obtain enough matches based on rules at (1), we will leverage the distributions of labeled jobs across the three categories above to infer the depth of the hierarchy  
    - Else, between firm approach is needed: 
        - Perform task (1) on the whole sample (across firms) to label jobs into 3 categories (high, middle, low). Through this step, we will extract our training sample.
        - Identify predictors of labels from training sample to classify test sample

3. Open questions:  
    - Given that every (or at least most) of the firms will have 2 layers, and we can only pick up 3 with this method, most likely the variation we can generate won’t be enough to observe any meaningful differences across firms. However, within companies: Can we leverage variation in the distribution of jobs within the 3 categories to infer something about the hierarchy? E.g., assuming firm A and B have both 100 employees, and A has 50 in middle category, while B has only 2, does this difference have any implications for the hierarchy of the two firms? 
    - How do we separate interactions of a job post with other jobs within the firm vs. outside the firm (e.g., customers or partners)? And between things and people, e.g. ‘keeps an eye on new technologies’ vs. ‘keeps an eye on subordinates’. 
        - There is also some question about identifying ‘subordinates’ and ‘colleagues on the same level/above’. For instance, ‘resolving conflicts’ could be a cool measure in line with Phanish’s model for hierarchy, but only if we know it’s about resolving conflicts at a level ‘below’, not at one’s own level (let alone with clients/partners outside the firm). 
    - How do we account for homonyms? 
        - One solution could be tagging. E.g., “reports to” (verb) belongs to our up-ward looking dictionary, and should be differentiated from “reports to” (noun) as in “writes reports to”, which is beyond our scope
    - It seems there is a lack of specific keywords for the up-ward looking dictionary--most of them are going to be the passive form of the keywords in the down-ward looking dictionary (?)  
        - On second thought, I wonder if this may be ok, in the sense that what we are really after is distinguishing between different levels of the managerial hierarchy, rather than distinguishing among different types of individual contributors. And it seems to me that if the job has any people management involved, it will mention that in the ad, so perhaps it’s reasonable to assume that if it doesn’t, it’s an individual contributor?
    - How do we validate that the labelling (task (2)) produces decent results? 
        - Projecting job adds into embedding space (based on cosine similarity of text). We hope to observe clusters overlapping with our labels induced via keyword search. 
        - Exporting a random sample of job titles associated with each category and having MTurkers/others judge whether those titles conform with our predictions (high-mid-low levels).

**Limitations of this approach:**
- Only 3 hierarchical levels
- Seems not bullet-proof on the subjectivity dimension. Despite starting from O*NET to generate the dictionaries, judgement calls & subjective choices in the identification of words remain.
