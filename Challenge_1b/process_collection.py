import os
import json
import glob
import fitz
import spacy
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Load SpaCy for summaries (lightweight)
nlp = spacy.load("en_core_web_sm")

# Load Gemma 3:1B (quantized MiniLM variant via HuggingFace/SBERT)
# This uses a SentenceTransformer wrapper for simplicity
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  
# NOTE: Replace with a Gemma 3:1B embedding model when available in your environment.

def dict_to_str(d):
    if isinstance(d, dict):
        return " ".join(str(v) for _, v in d.items())
    return str(d)

def extract_sections(pdf_path):
    """Extract section candidates (title + content) from a PDF."""
    doc = fitz.open(pdf_path)
    sections = []
    for idx, page in enumerate(doc):
        page_num = idx + 1
        blocks = page.get_text("dict")["blocks"]

        full_text, headings = "", []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    txt = span.get("text", "").strip()
                    if not txt:
                        continue
                    # Detect likely section headings
                    if txt.isupper() or txt.endswith(":") or span["size"] > 12:
                        if len(txt.split()) > 1:
                            headings.append(txt)
                    full_text += txt + " "

        if full_text.strip():
            title = headings[0] if headings else "Page Content"
            sections.append({
                "document": os.path.basename(pdf_path),
                "page_number": page_num,
                "section_title": title,
                "content": full_text.strip()
            })
    return sections

def embed_texts(texts):
    """Generate embeddings for a list of texts using Gemma or MiniLM."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def rank_sections(sections, query, top_n=5):
    """Rank sections by semantic similarity with the query."""
    section_texts = [s["content"] for s in sections]
    section_embeddings = embed_texts(section_texts)
    query_embedding = embed_texts([query])[0]

    sims = np.dot(section_embeddings, query_embedding)
    for i, score in enumerate(sims):
        sections[i]["score"] = float(score)

    ranked = sorted(sections, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]

def summarize_text(text):
    """Take first 3 sentences as a summary."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents][:3]
    return " ".join(sentences).strip()

def process_collection(collection_path):
    input_json = os.path.join(collection_path, "challenge1b_input.json")
    output_json = os.path.join(collection_path, "challenge1b_output.json")

    with open(input_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    persona = dict_to_str(config.get("persona", {}))
    job = dict_to_str(config.get("job_to_be_done", {}))
    query = f"{persona} {job}"

    pdf_dir = os.path.join(collection_path, "PDFs")
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    all_sections = []
    for pdf in pdf_files:
        all_sections.extend(extract_sections(pdf))

    # Rank top 5 using semantic similarity
    top_sections = rank_sections(all_sections, query, top_n=5)

    # Build final output
    result = {
        "metadata": {
            "input_documents": [os.path.basename(f) for f in pdf_files],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, sec in enumerate(top_sections, start=1):
        result["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": rank,
            "page_number": sec["page_number"]
        })
        result["subsection_analysis"].append({
            "document": sec["document"],
            "refined_text": summarize_text(sec["content"]),
            "page_number": sec["page_number"]
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for collection in os.listdir(base_dir):
        path = os.path.join(base_dir, collection)
        if os.path.isdir(path) and "Collection" in collection:
            print(f"Processing {collection}...")
            process_collection(path)
