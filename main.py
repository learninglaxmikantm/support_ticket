import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Read raw CSV (no header)
df_raw = pd.read_csv("support_tickets.csv", header=None)

# Split into columns
df = df_raw[0].str.replace('"', '').str.split(",", expand=True)

df.columns = [
    "ticket_id",
    "emp_id",
    "department",
    "business_unit",
    "ticket_date",
    "ticket_text",
    "category",
    "priority"
]

# ---------- PRIORITY MODEL ----------
priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=300
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

priority_model.fit(df["ticket_text"], df["priority"])
joblib.dump(priority_model, "priority.joblib")

# ---------- CATEGORY MODEL ----------
category_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=300
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

category_model.fit(df["ticket_text"], df["category"])
joblib.dump(category_model, "category.joblib")

print("Models saved using joblib")
