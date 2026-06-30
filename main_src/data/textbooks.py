from pathlib import Path
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent.parent

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

if __name__ == "__main__":

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fdir = PROJECT_DIR / "corpus" / "textbooks" / "en"
    fnames = sorted([fname.name for fname in fdir.iterdir()])

    if not (PROJECT_DIR / "corpus" / "textbooks" / "chunk").exists():
        (PROJECT_DIR / "corpus" / "textbooks" / "chunk").mkdir(parents=True, exist_ok=True)
        
    for fname in tqdm.tqdm(fnames):
        fpath = PROJECT_DIR / "corpus" / "textbooks" / "en" / fname
        texts = text_splitter.split_text(open(fpath).read().strip())
        saved_text = [json.dumps({"id": '_'.join([fname.replace(".txt", ''), str(i)]), "title": fname.strip(".txt"), "content": re.sub("\s+", " ", texts[i]), "contents": concat(fname.strip(".txt"), re.sub("\s+", " ", texts[i]))}) for i in range(len(texts))]
        with open(PROJECT_DIR / "corpus" / "textbooks" / "chunk" / fname.replace(".txt", ".jsonl"), 'w') as f:
            f.write('\n'.join(saved_text))