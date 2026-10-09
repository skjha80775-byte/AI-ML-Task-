# Task 5: Decision Trees and Random Forests

AI & ML Internship, Elevate Labs.

## Objective
Learn tree-based models for classification: Decision Tree, Random Forest, feature importance and cross-validation.

## Dataset
Heart Disease dataset (`heart.csv`, binary target: `target` / `HeartDisease`).
Place `heart.csv` in the same folder as `task5.py`.

## What I did
1. Trained a **Decision Tree Classifier** and visualized it (`plot_tree` and **Graphviz**).
2. **Overfitting analysis:** compared train, test and 5-fold CV accuracy for `max_depth` 1 to 20, then picked the best depth by CV.
3. Trained a **Random Forest** (200 trees) and compared it with the single tree.
4. **Feature importance:** Gini importance and permutation importance.
5. **Cross-validation:** 5-fold stratified CV for all models.

## How to run
```bash
pip install -r requirements.txt
# Graphviz binary is also needed for rendering (https://graphviz.org/download/)
python task5.py
```
Outputs are saved in `images/`: depth curve, tree plots, confusion matrices, feature importances, CV comparison, `cv_results.csv`.

## Results
Fill in after running (numbers are printed in the terminal):

| Model | Test acc | 5-fold CV acc |
|---|---|---|
| Decision Tree (unrestricted) | | |
| Decision Tree (best depth) | | |
| Random Forest (200 trees) | | |

**Key findings**
- The unrestricted tree reaches 100% train accuracy but lower test accuracy, which is overfitting. Limiting depth reduces the gap.
- Random Forest gives higher and more stable accuracy than a single tree because averaging many decorrelated trees lowers variance.
- Top features: (write the top 3 from the output, e.g. `cp`, `thalach`, `oldpeak`).
