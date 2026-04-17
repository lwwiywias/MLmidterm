# Star Wars Species Classifier; ML Midterm Project

Can a machine learning model tell a Human from an alien using only physical measurements?

This project builds an end-to-end ML pipeline on Star Wars character data from SWAPI. We train classifiers to predict whether a character is Human or Non-Human, segment characters into physical archetypes via clustering, and boost performance with ensemble methods.

## Dataset

| Property | Details |
|---|---|
| **Name** | Star Wars Characters (SWAPI-derived) |
| **Source** | https://www.kaggle.com/datasets/jsphyg/star-wars |
| **License** | CC0: Public Domain |
| **Size** | ~555 characters, 10 features |


## Installation & Running Order

```bash

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset - place in data/raw/characters.csv

# 3. Run notebooks in order
jupyter notebook notebooks/T1_EDA.ipynb
jupyter notebook notebooks/T2_Supervised.ipynb
jupyter notebook notebooks/T3_Unsupervised.ipynb
jupyter notebook notebooks/T4_Ensemble.ipynb
```

## Final Model Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| T2 — Decision Tree (baseline) | 0.896 | 0.744 | 1.000 | 0.853 |
| T4 — Random Forest | 0.915 | 0.871 | 0.844	| 0.857 |
| T4 — Gradient Boosting | .925 |	0.875 |	0.875	| 0.875 |


## Repository Structure

```
starwars-ml/
  data/
    raw/            
    cleaned.csv     
    clustered.csv   
  notebooks/
    T1_EDA.ipynb
    T2_Supervised.ipynb
    T3_Unsupervised.ipynb
    T4_Ensemble.ipynb
  models/
    supervised_best.pkl
  reports/          
  requirements.txt
  README.md
```

## Key Finding from EDA

 One of the notable findinds is that eye_color was accounted for nearly 48% of the model's decision-making. This is scientifically explainable in-universe (alien species have distinct eye colours by design), and the eye colour bar chart from Task 1 visually confirms it

