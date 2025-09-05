import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os 


st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")

@st.cache_resource
def load_kb_and_vectorizer(csv_path: str):
    # Load KB
    df = pd.read_csv(csv_path)
    # Basic cleanup
    df["question"] = df["question"].fillna("").str.strip()
    df["answer"] = df["answer"].fillna("").str.strip()
    # Vectorizer fit
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["question"])
    return df, vectorizer, X

def best_match_answer(user_text: str, df: pd.DataFrame, vectorizer, X, threshold: float = 0.25):
    vec = vectorizer.transform([user_text])
    sims = cosine_similarity(vec, X).flatten()
    idx = sims.argmax()
    score = float(sims[idx])
    if score >= threshold:
        return df.iloc[idx]["answer"], score, df.iloc[idx]["question"]
    return None, 0.0, None

def detect_and_store_name(user_text: str):
    # Very small NER-esque rule: "my name is <name>" or "I am <name>" or "I'm <name>"
    patterns = [
        r"\bmy\s+name\s+is\s+([A-Z][a-zA-Z\-']+)\b",
        r"\bi\s*am\s+([A-Z][a-zA-Z\-']+)\b",
        r"\bi'm\s+([A-Z][a-zA-Z\-']+)\b",
    ]
    for p in patterns:
        m = re.search(p, user_text, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            st.session_state["user_name"] = name
            return name
    return None

def small_talk(user_text: str):
    txt = user_text.lower().strip()
    if any(g in txt for g in ["hello", "hi", "hey"]):
        name = st.session_state.get("user_name")
        return f"Hey{' ' + name if name else ''}! How can I help today?"
    if "thank" in txt:
        return "You're welcome!"
    if any(x in txt for x in ["bye", "goodbye", "see you"]):
        return "Bye! Have a great day 👋"
    if "help" in txt or "what can you do" in txt:
        return "I can answer questions from my small knowledge base. Try asking about 'project', 'how it works', or 'tech stack'."
    return None

st.title("🤖 Simple Chatbot (Local Retrieval)")

kb_path = "knowledge_base.csv"
if not os.path.exists(kb_path):
    st.error("knowledge_base.csv not found. Please keep it next to app.py.")
    st.stop()

df_kb, vectorizer, X = load_kb_and_vectorizer(kb_path)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.25, 0.01)
    st.caption("Higher = stricter matching from the knowledge base.")
    if st.button("Reset chat"):
        st.session_state["messages"] = []
        st.experimental_rerun()
    st.markdown("---")
    st.markdown("**Tip:** Edit `knowledge_base.csv` to customize the bot's answers and then **Rerun** the app.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a tiny chatbot. Ask me anything about this demo project."}
    ]

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m
                    ["content"])

user_text = st.chat_input("Type your message...")
if user_text:
    # Save user turn
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Name capture
    name = detect_and_store_name(user_text)

    # Try KB retrieval
    answer, score, matched_q = best_match_answer(user_text, df_kb, vectorizer, X, threshold=threshold)

    # Fallback: small talk
    if not answer:
        answer = small_talk(user_text)

    # Final fallback
    if not answer:
        topics = ", ".join(sorted(set(df_kb["question"].str.split().str[0].dropna())))[:120]
        answer = "I'm still learning. Try asking me something found in my knowledge base or rephrase your question."

    # Personalize if name known
    if st.session_state.get("user_name"):
        answer = f"{st.session_state['user_name']}, {answer}"

    with st.chat_message("assistant"):
        if matched_q:
            st.markdown(answer)
            st.caption(f"(Matched: \"{matched_q}\" • score={score:.2f})")
        else:
            st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
