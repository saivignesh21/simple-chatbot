# Simple Chatbot (Local Retrieval)

A tiny, beginner-friendly chatbot that answers from a CSV knowledge base using TF‑IDF (no external APIs). Built with **Python + Streamlit + scikit-learn**.

## 🚀 Quickstart

```bash
# 1) Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Open the URL that Streamlit prints (usually https://simple-chatbot-7app4b72tznyzhberegujff.streamlit.app/).

## 🧠 Customize the knowledge base

Open `knowledge_base.csv` and add your own Q/A rows:

```csv
question,answer
How do refunds work?,Refunds are processed within 5-7 business days to the original payment method.
```

Save the file and **Rerun** the app.

## 🧩 How it works

- We vectorize all questions in the CSV with **TF‑IDF**.
- For each user query, we compute cosine similarity against those vectors.
- If the top score is above a threshold (default **0.25**, configurable in the sidebar), we return the paired answer.

## 🧪 Tips

- Keep questions short and specific.
- Add multiple variants of the same question for better matching.
- Adjust the threshold slider if answers feel too loose or too strict.

## 🐙 Push to GitHub (one-time setup)

1. Create a **new empty repo** on GitHub (no README/license).
2. In this project folder, run:

```bash
git init
git add .
git commit -m "Initial commit: Simple Chatbot (Streamlit + TF-IDF)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Next time you make changes:
> ```bash
> git add .
> git commit -m "Update"
> git push
> ```

## 📦 Project layout

```
simple-chatbot/
├─ app.py
├─ knowledge_base.csv
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## ✅ Next steps (optional)

- Add a sidebar form to **append Q/A** pairs and write them back to the CSV.
- Add **conversation memory** beyond the tiny name-capture rule.
- Integrate an LLM API for fallback answers.
- Deploy on **Streamlit Community Cloud**, **Railway**, **Render**, or **Fly.io**.
