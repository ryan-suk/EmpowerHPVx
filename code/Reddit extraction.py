import requests
import pandas as pd

# 1. List of Reddit threads (use the JSON endpoints)
urls = [
    'https://www.reddit.com/r/Millennials/comments/1av96vn/.json',
    'https://www.reddit.com/r/VACCINES/comments/1gt0j30/.json',
    'https://www.reddit.com/r/Xennials/comments/1ceqs67/.json',
    'https://www.reddit.com/r/HPV/comments/18r8ylc/.json',
    'https://www.reddit.com/r/AskGaybrosOver30/comments/170bnxw/.json',
    # …add as many as you like
]

headers = {'User-Agent': 'CommentExtractor/0.1'}
rows = []

for url in urls:
    resp = requests.get(url, headers=headers)
    data = resp.json()

    # Extract OP text
    post = data[0]['data']['children'][0]['data']
    title = post.get('title', '')
    body = post.get('selftext', '')
    op_text = title + ("\n\n" + body if body else "")

    # Extract just the comment bodies in order
    comments = []


    def extract_bodies(c):
        d = c['data']
        comments.append(d.get('body', ''))
        replies = d.get('replies')
        if isinstance(replies, dict):
            for child in replies['data']['children']:
                if child.get('kind') == 't1':
                    extract_bodies(child)


    for child in data[1]['data']['children']:
        if child.get('kind') == 't1':
            extract_bodies(child)

    # Build a single-row dict
    row = {'op': op_text}
    for idx, text in enumerate(comments, start=1):
        row[f'comment{idx}'] = text

    rows.append(row)

# 3. Convert to DataFrame (Pandas will auto-align columns across rows)
df = pd.DataFrame(rows)

# 4. Save to Excel
df.to_excel('reddit_threads_compiled.xlsx', index=False)

print(f"Wrote {len(rows)} threads with up to {df.shape[1] - 1} comments each.")
