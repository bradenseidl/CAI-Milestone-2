# Milestone 2: Reproduction of Bishi et al. (2024)

This project reproduces the SMS spam detection models from the paper  
**“Optimizing SMS Spam Detection: Leveraging the Strength of a Voting Classifier Ensemble”**  
by *Bishi, M. R., Manikanta, N. S., Bharadwaj, G. H. S., Teja, P. S. K., & Rao, G. R. K.* (2024).

The `milestone2.py` script runs the complete experiment including preprocessing (tokenization, stop word removal, stemming), model training, evaluation, and statistical comparison.

---

## How to Reproduce

### 1. Clone the Repository

```bash
git clone https://github.com/bradenseidl/CAI-Milestone-2.git
cd CAI-Milestone-2
```

---

### 2. Environment Setup

**Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**Install required libraries:**
```bash
pip install -r requirements.txt
```

---

### 3. Dataset

**Download the dataset from the UCI Machine Learning Repository:**  
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Setup the data folder:**
```bash
mkdir data
```

Place the file `SMSSpamCollection` inside the `data/` directory.

---

### 4. Run the Experiment

The code will automatically download NLTK tokenizer data (`punkt`) if missing.

```bash
python milestone2.py
```

---

### 5. Output

- The script prints model performance metrics (**Accuracy**, **Precision**, **Recall**, **F1**).  
- Includes 5-fold cross-validation results for stability.  
- Performs a one-sample **t-test** comparing Extra Trees accuracy against the paper’s reported results.  
- Generates ROC and Precision–Recall plots.

All plots are automatically saved to a `reports/` folder.

```bash
mkdir reports   # only needed once
```

---

## Reproducibility Notes

- **Random Seed:** `RANDOM_STATE = 42` ensures deterministic splits and results.  
- **Libraries:** see `requirements.txt` for full environment details.  
- **Outputs:** ROC and PR curves saved in `/reports`.  

---

Minor implementation assumptions (e.g., default TF–IDF parameters, 300 estimators for tree models) are noted in the accompanying report.
