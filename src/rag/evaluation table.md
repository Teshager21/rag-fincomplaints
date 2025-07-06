# RAG Pipeline Evaluation

| Question                          | Generated Answer                                            | Retrieved Sources (Sample)                                                                                          | Quality Score (1-5) | Comments / Analysis                                                                                     |
|----------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------|
| 1. Why are there hidden fees?    | Because they cheated me out of many hundreds of dollars     | 1. deceptive practice with hidden condition, no warning<br>2. hidden service fees and inflated pricing<br>3. lack of transparency | 4                   | Answer is relevant and supported by retrieved chunks showing deceptive hidden fees and financial harm.  |
| 2. Why do companies delay payments? | Because the company does not require such an unnecessary delay before payments may be applied --- | 1. payment processing causing delays<br>2. vague account agreement<br>3. delay harms proactive payers | 4                   | Answer matches retrieved chunks well, explaining delays and consumer frustration with payment timing.   |
| 3. What issues occur with BNPL products? | 1. dissatisfaction with the product the product did not meet promised standards or my expectations, and i quickly canceled as a result. 2 --- | 1. growth of BNPL products<br>2. dissatisfaction with product<br>3. customer rights concerns | 4                   | Good summary of dissatisfaction and issues customers face with BNPL products as per context chunks.      |
| 4. How do consumers describe Truist Bank? | Consumers are not misled by Truist Bank.                    | 1. consumers need help against banks<br>2. complaints about practices<br>3. concern over bank's treatment of consumers | 2                   | Generated answer contradicts context; retrieved chunks indicate negative consumer views and complaints. |
| 5. How often do people mention fraud in credit cards? | a handful of cases of fraud occurred on my cards           | 1. fraud on credit cards<br>2. limited fraud cases over 30 years<br>3. identity theft reports | 5                   | Clear, concise answer well supported by retrieved context mentioning specific fraud instances.          |

---

### Notes:
- Scores are subjective based on relevance, completeness, and consistency between answer and retrieved context.
- Question 4’s answer could be improved; consider prompt tuning or better filtering of retrieved chunks.
- The pipeline generally produces strong, context-supported answers for factual questions.
