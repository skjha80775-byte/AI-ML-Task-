"""
Task 5: Decision Trees and Random Forests
Dataset: Heart Disease (heart.csv, target column: 'target' or 'HeartDisease')
If heart.csv is not found, falls back to sklearn's breast-cancer dataset
so the script still runs end to end.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = "images"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 1. Load data
def load_data():
    if os.path.exists("heart.csv"):
        df = pd.read_csv("heart.csv")
        target = "target" if "target" in df.columns else "HeartDisease"
        df = pd.get_dummies(df, drop_first=True)  # no-op if all numeric
        X, y = df.drop(columns=[target]), df[target]
        name = "Heart Disease"
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        name = "Breast Cancer (fallback, heart.csv not found)"
    return X, y, name


X, y, dataset_name = load_data()
print(f"Dataset: {dataset_name} | shape: {X.shape} | class balance:\n{y.value_counts(normalize=True).round(3)}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
class_names = [str(c) for c in sorted(y.unique())]

# ------------------------------------------- 2. Decision Tree (unrestricted)
dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
print("=== Decision Tree (no depth limit) ===")
print(f"Depth: {dt_full.get_depth()} | Leaves: {dt_full.get_n_leaves()}")
print(f"Train acc: {dt_full.score(X_train, y_train):.3f} | Test acc: {dt_full.score(X_test, y_test):.3f}\n")

# ------------------------------- 3. Overfitting analysis: vary max_depth
depths = range(1, 21)
train_acc, test_acc, cv_acc = [], [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
    train_acc.append(m.score(X_train, y_train))
    test_acc.append(m.score(X_test, y_test))
    cv_acc.append(cross_val_score(m, X_train, y_train, cv=5).mean())

best_depth = list(depths)[int(np.argmax(cv_acc))]
print(f"Best max_depth by 5-fold CV: {best_depth} (CV acc = {max(cv_acc):.3f})\n")

plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, "o-", label="Train")
plt.plot(depths, test_acc, "s-", label="Test")
plt.plot(depths, cv_acc, "^-", label="5-fold CV")
plt.axvline(best_depth, color="grey", ls="--", label=f"best depth = {best_depth}")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Overfitting analysis: accuracy vs tree depth")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/overfitting_depth_curve.png", dpi=150); plt.close()

# ------------------------------------ 4. Pruned tree + visualization
dt = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
print(f"=== Decision Tree (max_depth={best_depth}) ===")
print(f"Train acc: {dt.score(X_train, y_train):.3f} | Test acc: {dt.score(X_test, y_test):.3f}\n")

# (a) matplotlib
plt.figure(figsize=(22, 10))
plot_tree(dt, feature_names=X.columns, class_names=class_names, filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title(f"Decision Tree (max_depth={best_depth}, top 3 levels shown)")
plt.savefig(f"{OUT}/decision_tree.png", dpi=150, bbox_inches="tight"); plt.close()

# (b) Graphviz
export_graphviz(dt, out_file=f"{OUT}/tree.dot", feature_names=X.columns, class_names=class_names,
                filled=True, rounded=True, special_characters=True, max_depth=3)
try:
    import graphviz
    graphviz.Source.from_file(f"{OUT}/tree.dot").render(f"{OUT}/tree_graphviz", format="png", cleanup=True)
except Exception as e:
    print("Graphviz render skipped:", e)

with open(f"{OUT}/tree_rules.txt", "w") as f:
    f.write(export_text(dt, feature_names=list(X.columns)))

# --------------------------------------------------- 5. Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("=== Random Forest (200 trees) ===")
print(f"Train acc: {rf.score(X_train, y_train):.3f} | Test acc: {rf.score(X_test, y_test):.3f}")
print(classification_report(y_test, rf_pred))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test, ax=ax[0], colorbar=False)
ax[0].set_title("Decision Tree")
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[1], colorbar=False)
ax[1].set_title("Random Forest")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150); plt.close()

# ------------------------------------------- 6. Feature importances
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
top = imp.tail(15)
plt.figure(figsize=(8, 6))
top.plot(kind="barh", color="#4c72b0")
plt.title("Random Forest feature importances (Gini, top 15)")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
perm_s.tail(15).plot(kind="barh", color="#dd8452")
plt.title("Permutation importance on test set (top 15)")
plt.xlabel("Mean accuracy drop"); plt.tight_layout()
plt.savefig(f"{OUT}/permutation_importance.png", dpi=150); plt.close()

print("Top 5 features (Gini):\n", imp.sort_values(ascending=False).head(5).round(4), "\n")
print("Top 5 features (permutation):\n", perm_s.sort_values(ascending=False).head(5).round(4), "\n")

# --------------------------------------- 7. Cross-validation comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {
    "Decision Tree (unrestricted)": DecisionTreeClassifier(random_state=RANDOM_STATE),
    f"Decision Tree (max_depth={best_depth})": DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
}
rows = []
for name, model in results.items():
    scores = cross_val_score(model, X, y, cv=cv)
    rows.append({"Model": name, "CV mean acc": scores.mean(), "CV std": scores.std()})
summary = pd.DataFrame(rows).round(4)
print("=== 5-fold Cross-validation (full data) ===")
print(summary.to_string(index=False))
summary.to_csv(f"{OUT}/cv_results.csv", index=False)

plt.figure(figsize=(7, 4))
plt.bar(summary["Model"], summary["CV mean acc"], yerr=summary["CV std"], capsize=5,
        color=["#c44e52", "#dd8452", "#55a868"])
plt.ylim(max(0, summary["CV mean acc"].min() - 0.1), 1.0)
plt.ylabel("CV accuracy"); plt.xticks(rotation=15, ha="right"); plt.title("Cross-validation comparison")
plt.tight_layout(); plt.savefig(f"{OUT}/cv_comparison.png", dpi=150); plt.close()
print(f"\nAll plots saved in ./{OUT}/")
