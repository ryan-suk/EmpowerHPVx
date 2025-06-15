import os
import time
import json
import pandas as pd
import openai

# **Ensure you have openai installed**:
# pip install openai

# 1. Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or hardcode your token

# 2. Load the dataset
df = pd.read_excel(r'C:\postdata.xlsx')

# 3. Combine all threads into one large corpus
threads = []
original_col = "Original_post"
comment_cols = [c for c in df.columns if c.startswith("Comment_")]

for idx, row in df.iterrows():
    orig = str(row[original_col])
    comments = "\n".join(str(row[c]) for c in comment_cols if pd.notna(row[c]))
    threads.append(f"Thread {idx}:\nOriginal:\n{orig}\nComments:\n{comments}")

full_corpus = "\n\n".join(threads)

# 4. Helpers for chunking & summarization
MAX_WORDS_BEFORE_SUMMARY = 2000
CHUNK_SIZE = 1000
SUMMARY_MAX_TOKENS = 500

def count_words(text: str) -> int:
    return len(text.split())

def chunk_text(text: str, size: int) -> list:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def summarize_text(text: str) -> str:
    prompt = (
        "Summarize this text into up to 3 concise paragraphs, preserving key points and "
        "including concerns and perception:\n\n" + text
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise summarizer, with HPV expertise."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=SUMMARY_MAX_TOKENS
    )
    return resp.choices[0].message.content.strip()

def condense_corpus(text: str) -> str:
    """Chunk & summarize the full corpus to fit within context limits."""
    if count_words(text) <= MAX_WORDS_BEFORE_SUMMARY:
        return text
    chunks = chunk_text(text, CHUNK_SIZE)
    summaries = []
    for chunk in chunks:
        time.sleep(1)  # rate-limit guard
        summaries.append(summarize_text(chunk))
    merged = "\n\n".join(summaries)
    if count_words(merged) > MAX_WORDS_BEFORE_SUMMARY:
        return summarize_text(merged)
    return merged

# 5. Condense if needed
condensed_corpus = condense_corpus(full_corpus)

# 6. Single LLM call for overall analysis, with added items
analysis_prompt = (
    "You are a content analysis and NLP assistant. For the entire dataset of threads and comments, extract:\n"
    "1. Key themes (list and probability for emerging in the data set).\n"
    "2. Common questions that emerged (list and probability for emerging in the data set).\n"
    "3. Most commonly emerging misconceptions related to HPV and vaccine (list).\n\n"
    f"Dataset summary or threads:\n\"\"\"{condensed_corpus}\"\"\"\n\n"
    "Respond only in valid JSON with keys: themes, questions, misconceptions."
)
resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert in thematic analysis and natural language processing, with HPV expertise."},
        {"role": "user",   "content": analysis_prompt}
    ],
    temperature=0.0
)

# 7. Parse JSON response
raw = resp.choices[0].message.content
start, end = raw.find("{"), raw.rfind("}") + 1
parsed = json.loads(raw[start:end])

# 8. Output the overall results
print("Global Key Themes:", parsed.get("themes", []))
print("\nGlobal Common Questions:", parsed.get("questions", []))
print("\nMisconceptions:", parsed.get("misconceptions", []))

# 9. Save to JSON
with open("global_analysis100.json", "w") as f:
    json.dump(parsed, f, indent=2)

print("\nOverall analysis saved to global_analysis100.json")
