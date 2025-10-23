# Milestone 2: Reproduction of Bishi et al. (2024) on SMS Spam Detection 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

RANDOM_STATE = 42

rows = []
with open("data/SMSSpamCollection", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        rows.append((parts[0], parts[1]))

df = pd.DataFrame(rows, columns=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].astype(str).str.strip()

print(f"(shape, counts) after load: {df.shape} {df['label'].map({0:'ham',1:'spam'}).value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].values,
    df["label"].values,
    test_size=0.20,
    stratify=df["label"].values,
    random_state=RANDOM_STATE,
)

# download the 'punkt' tokenizer models
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# download the 'punkt_tab' resource needed by the tokenizer
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class StemmingTokenizer:
    """Custom tokenizer that also stems words."""
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(doc)]

# TF-IDF 
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    min_df=2,
    tokenizer=StemmingTokenizer() 
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Models
nb  = MultinomialNB()
svm = LinearSVC(random_state=RANDOM_STATE)
rf  = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
et  = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

# NB + SVM + RF + ET
voter = VotingClassifier(
    estimators=[("nb", nb), ("svm", svm), ("rf", rf), ("et", et)],
    voting="hard"
)

def evaluate(name, clf):
    clf.fit(X_train_tfidf, y_train)
    pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name}\nAcc:{acc:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1:.4f}")
    print("Confusion matrix:\n", cm)
    return name, acc, prec, rec, f1

# Fit + evaluate
rows_out = [
    evaluate("Naive Bayes", nb),
    evaluate("Linear SVM", svm),
    evaluate("Random Forest", rf),
    evaluate("Extra Trees", et),
    evaluate("Voting (NB+SVM+RF+ET)", voter)
]

res = pd.DataFrame(rows_out, columns=["model","accuracy","precision","recall","f1"]).sort_values("f1", ascending=False)
print("\nSummary:\n", res.to_string(index=False))

# 5-fold CV on TRAIN for stability (F1)
for name, clf in [("NB", nb), ("SVM", svm), ("RF", rf), ("ET", et), ("VOTE", voter)]:
    scores_f1 = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)
    print(f"{name}  CV F1 (mean±std): {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")

# Statistical Hypothesis Testing (Extra Trees vs paper)
print("\n--- Statistical Hypothesis Testing ---")
paper_accuracy_etc = 0.977756  # Table 4 (Extra Trees)

etc_accuracy_scores = cross_val_score(
    et, X_train_tfidf, y_train, cv=5, scoring="accuracy", n_jobs=-1
)

print(f"\nMy 5 CV Accuracy Scores for Extra Trees: {np.round(etc_accuracy_scores, 6)}")
print(f"My Mean CV Accuracy: {etc_accuracy_scores.mean():.6f}")
print(f"My Std  CV Accuracy: {etc_accuracy_scores.std():.6f}")
print(f"My Accuracy Range : {etc_accuracy_scores.min():.6f} – {etc_accuracy_scores.max():.6f}")
print(f"Paper's Reported Accuracy: {paper_accuracy_etc}")

t_statistic, p_value = stats.ttest_1samp(a=etc_accuracy_scores, popmean=paper_accuracy_etc)
print(f"\nT-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("\nConclusion: The p-value is > 0.05. We fail to reject the null hypothesis.")
    print("This means there is no statistically significant difference between my model's performance and the paper's.")
else:
    print("\nConclusion: The p-value is <= 0.05. We reject the null hypothesis.")
    print("This means there is a statistically significant difference between my model's performance and the paper's.")

# AUROC & Precision–Recall 
svm_cal = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=5)
soft_voter = VotingClassifier(
    estimators=[
        ("nb", MultinomialNB()),
        ("svm_cal", svm_cal),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE))
    ],
    voting="soft"
)
soft_voter.fit(X_train_tfidf, y_train)

y_scores = soft_voter.predict_proba(X_test_tfidf)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Precision–Recall
precisions, recalls, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recalls, precisions)

print(f"\nAUROC (Voting, soft): {roc_auc:.4f}")
print(f"PR  AUC (Voting, soft): {pr_auc:.4f}")

# Save plots
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – VotingClassifier (Soft, NB+SVM+RF+ET)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=200)
print("Saved: reports/roc_curve.png")

plt.figure()
plt.plot(recalls, precisions, label=f"PR (AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – VotingClassifier (Soft, NB+SVM+RF+ET)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("reports/pr_curve.png", dpi=200)
print("Saved: reports/pr_curve.png")