import os
import json
import glob
import fitz
import spacy
import numpy as np
from datetime import datetime
import ollama
import re

# Load SpaCy for sentence segmentation (lightweight, CPU-friendly)
nlp = spacy.load("en_core_web_sm")

# --- Check available models ---
def check_available_models():
    """Check which embedding models are available in Ollama"""
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        print(f"Available Ollama models: {available_models}")
        return available_models
    except Exception as e:
        print(f"Error checking available models: {e}")
        return []

# --- Improved fallback embedding using TF-IDF-like approach ---
def fallback_embedding(text: str, dim=384) -> np.ndarray:
    """Create a more sophisticated text-based embedding when Ollama embeddings fail"""
    import re
    from collections import Counter
    
    # Clean and tokenize text
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text_clean.split()
    
    # Create feature vector
    features = np.zeros(dim)
    
    if not words:
        return features.astype(np.float32)
    
    # Get word frequencies
    word_freq = Counter(words)
    total_words = len(words)
    
    # Important keywords for travel planning
    travel_keywords = {
        'city', 'cities', 'destination', 'location', 'place', 'visit', 'explore',
        'restaurant', 'hotel', 'accommodation', 'stay', 'dining', 'eat', 'food',
        'activity', 'activities', 'adventure', 'beach', 'attraction', 'tour',
        'culture', 'history', 'tradition', 'museum', 'art', 'heritage',
        'nightlife', 'bar', 'club', 'entertainment', 'music', 'dance',
        'travel', 'trip', 'vacation', 'holiday', 'journey', 'planning',
        'tips', 'advice', 'guide', 'recommendation', 'suggestion',
        'group', 'friends', 'college', 'student', 'young', 'budget',
        'day', 'days', 'itinerary', 'schedule', 'time', 'duration'
    }
    
    # Cuisine-related keywords
    cuisine_keywords = {
        'cuisine', 'cooking', 'wine', 'taste', 'flavor', 'recipe', 'chef',
        'market', 'local', 'traditional', 'specialty', 'dish', 'meal'
    }
    
    # Feature extraction
    keyword_score = 0
    cuisine_score = 0
    
    for word, freq in word_freq.items():
        # Basic word hash features
        word_hash = hash(word) % (dim - 20)  # Reserve space for special features
        features[word_hash] += freq / total_words
        
        # Keyword scoring
        if word in travel_keywords:
            keyword_score += freq
        if word in cuisine_keywords:
            cuisine_score += freq
    
    # Add special features at the end
    features[-20] = len(words) / 100.0  # Text length
    features[-19] = len(set(words)) / len(words)  # Vocabulary diversity
    features[-18] = keyword_score / total_words  # Travel keyword density
    features[-17] = cuisine_score / total_words  # Cuisine keyword density
    
    # Add n-gram features for better context
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    for bigram in bigrams[:20]:  # Limit bigrams
        bigram_hash = hash(bigram) % (dim - 20)
        features[bigram_hash] += 1.0 / len(bigrams)
    
    # Text structure features
    features[-16] = text.count(':') / len(text)  # Colon density (headings)
    features[-15] = text.count('.') / len(text)  # Sentence density
    features[-14] = text.count('!') / len(text)  # Exclamation density
    features[-13] = len([w for w in words if w.isupper()]) / total_words  # Uppercase density
    
    # Content type indicators
    if any(keyword in text.lower() for keyword in ['restaurant', 'hotel', 'accommodation']):
        features[-12] = 1.0
    if any(keyword in text.lower() for keyword in ['beach', 'activity', 'adventure']):
        features[-11] = 1.0
    if any(keyword in text.lower() for keyword in ['nightlife', 'bar', 'entertainment']):
        features[-10] = 1.0
    if any(keyword in text.lower() for keyword in ['history', 'culture', 'tradition']):
        features[-9] = 1.0
    if any(keyword in text.lower() for keyword in ['tip', 'advice', 'packing']):
        features[-8] = 1.0
    
    # Normalize to unit vector
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features.astype(np.float32)

# Global flag to track if we've already tried embedding models
_embedding_model_available = None
_tried_embedding_models = False

# --- Embed a single text using Ollama (try multiple models) ---
def embed_text(text: str) -> np.ndarray:
    global _embedding_model_available, _tried_embedding_models
    
    # If we haven't tried embedding models yet, try them once
    if not _tried_embedding_models:
        embedding_models = [
            'nomic-embed-text',
            'all-minilm', 
            'mxbai-embed-large',
            'snowflake-arctic-embed',
            'bge-large'
        ]
        
        for model in embedding_models:
            try:
                res = ollama.embeddings(model=model, prompt="test")
                _embedding_model_available = model
                print(f"  Using embedding model: {model}")
                break
            except Exception:
                continue
        
        _tried_embedding_models = True
        if _embedding_model_available is None:
            print("  No embedding models available. Using fallback similarity method.")
    
    # Use the available embedding model or fallback
    if _embedding_model_available:
        try:
            res = ollama.embeddings(model=_embedding_model_available, prompt=text)
            return np.array(res["embedding"], dtype=np.float32)
        except Exception:
            # If the model fails, fall back
            _embedding_model_available = None
    
    return fallback_embedding(text)

# --- Embed a list of texts (with error handling) ---
def embed_texts(texts):
    embeddings = []
    print(f"  Generating embeddings for {len(texts)} sections...")
    
    for i, t in enumerate(texts):
        if i % 20 == 0:  # Less frequent progress updates
            print(f"    Progress: {i}/{len(texts)}")
        try:
            embeddings.append(embed_text(t))
        except Exception as e:
            print(f"    Error embedding text {i}: {e}")
            # Use fallback for this specific text
            embeddings.append(fallback_embedding(t))
    
    print(f"    Completed: {len(embeddings)}/{len(texts)}")
    return np.vstack(embeddings)

# --- Flatten persona/job dicts into strings ---
def dict_to_str(d):
    if isinstance(d, dict):
        return " ".join(str(v) for _, v in d.items())
    return str(d)

# --- Improved heading detection ---
def is_heading(span, text):
    """Detect if a text span is likely a heading"""
    if not text or len(text.strip()) < 3:
        return False
    
    # Check font size (headings are usually larger)
    font_size = span.get("size", 0)
    
    # Check if text looks like a heading
    text_clean = text.strip()
    
    # Strong indicators of headings
    if (font_size > 14 or  # Large font
        span.get("flags", 0) & 2**4 or  # Bold text
        text_clean.endswith(":") and len(text_clean.split()) <= 8 or  # Short text ending with colon
        text_clean.isupper() and len(text_clean.split()) <= 6 or  # Short uppercase text
        re.match(r'^[A-Z][a-z]+ [A-Za-z\s]+$', text_clean) and len(text_clean.split()) <= 8):  # Title case
        return True
    
    return False

# --- Extract text sections with improved heading detection ---
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    
    for idx, page in enumerate(doc):
        page_num = idx + 1
        blocks = page.get_text("dict")["blocks"]
        
        current_section = {"heading": None, "content": "", "page_num": page_num}
        page_headings = []
        page_content = []
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                line_spans = []
                
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if text:
                        line_text += text + " "
                        line_spans.append((span, text))
                
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # Check if this line contains a heading
                is_line_heading = False
                if line_spans:
                    # Use the first span to determine if it's a heading
                    first_span, first_text = line_spans[0]
                    if is_heading(first_span, line_text):
                        is_line_heading = True
                        page_headings.append(line_text)
                
                if not is_line_heading:
                    page_content.append(line_text)
        
        # Combine content into meaningful chunks
        full_content = " ".join(page_content)
        
        if full_content.strip():
            # Use the most relevant heading or create a descriptive title
            if page_headings:
                # Choose the longest/most descriptive heading
                title = max(page_headings, key=len)
            else:
                # Create a meaningful title based on content
                content_words = full_content.lower().split()
                if any(word in content_words for word in ['restaurant', 'hotel', 'accommodation']):
                    title = "Restaurants and Hotels"
                elif any(word in content_words for word in ['city', 'cities', 'town']):
                    title = "Cities and Destinations"
                elif any(word in content_words for word in ['food', 'cuisine', 'cooking', 'wine']):
                    title = "Culinary Experiences"
                elif any(word in content_words for word in ['activity', 'activities', 'adventure', 'beach']):
                    title = "Activities and Adventures"
                elif any(word in content_words for word in ['nightlife', 'bar', 'club', 'entertainment']):
                    title = "Nightlife and Entertainment"
                elif any(word in content_words for word in ['packing', 'tips', 'travel']):
                    title = "Travel Tips and Packing"
                elif any(word in content_words for word in ['history', 'culture', 'tradition']):
                    title = "History and Culture"
                else:
                    title = f"Page {page_num} Content"
            
            sections.append({
                "document": os.path.basename(pdf_path),
                "page_number": page_num,
                "section_title": title,
                "content": full_content.strip()
            })
    
    return sections

# --- Enhanced ranking with better similarity scoring ---
def rank_sections(sections, query, top_n=5):
    if not sections:
        return []
    
    # Create enhanced text for embedding (title + content)
    section_texts = []
    for s in sections:
        enhanced_text = f"{s['section_title']} {s['content']}"
        section_texts.append(enhanced_text)
    
    section_embeddings = embed_texts(section_texts)
    query_embedding = embed_text(query)
    
    # Normalize embeddings for better cosine similarity
    section_embeddings = section_embeddings / np.linalg.norm(section_embeddings, axis=1, keepdims=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Calculate cosine similarity
    similarities = np.dot(section_embeddings, query_embedding)
    
    # Add diversity bonus to avoid all results from same document
    document_counts = {}
    for i, section in enumerate(sections):
        doc_name = section["document"]
        document_counts[doc_name] = document_counts.get(doc_name, 0) + 1
    
    # Apply diversity scoring
    final_scores = []
    doc_appearance_count = {}
    
    for i, sim_score in enumerate(similarities):
        doc_name = sections[i]["document"]
        doc_appearance_count[doc_name] = doc_appearance_count.get(doc_name, 0) + 1
        
        # Reduce score for documents that appear too frequently
        diversity_penalty = 0.1 * doc_appearance_count[doc_name]
        final_score = sim_score - diversity_penalty
        
        sections[i]["score"] = float(final_score)
        final_scores.append((i, final_score))
    
    # Sort by final score
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top sections ensuring diversity
    result = []
    used_docs = []
    
    for idx, score in final_scores:
        if len(result) >= top_n:
            break
        
        doc_name = sections[idx]["document"]
        # Allow maximum 2 sections from same document
        if used_docs.count(doc_name) < 2:
            result.append(sections[idx])
            used_docs.append(doc_name)
    
    # If we don't have enough diverse results, fill with remaining top scores
    if len(result) < top_n:
        for idx, score in final_scores:
            if len(result) >= top_n:
                break
            if sections[idx] not in result:
                result.append(sections[idx])
    
    return result[:top_n]

# --- Improved text summarization ---
def summarize_text(text):
    """Create a better summary by taking meaningful sentences"""
    if not text:
        return ""
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    
    # Take first 4-5 sentences or up to 300 characters, whichever comes first
    summary_sentences = []
    char_count = 0
    
    for sentence in sentences[:5]:
        if char_count + len(sentence) > 500:  # Limit summary length
            break
        summary_sentences.append(sentence)
        char_count += len(sentence)
    
    if not summary_sentences and text:
        # Fallback: take first 300 characters
        return text[:300].strip() + ("..." if len(text) > 300 else "")
    
    return " ".join(summary_sentences).strip()

# --- Main pipeline for each collection ---
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
        print(f"  Processing {os.path.basename(pdf)}...")
        sections = extract_sections(pdf)
        all_sections.extend(sections)

    print(f"  Extracted {len(all_sections)} sections total")
    
    # Rank top 5 sections semantically
    top_sections = rank_sections(all_sections, query, top_n=5)

    # Prepare final output JSON
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

    print(f"  Results saved to {output_json}")

if __name__ == "__main__":
    # Check available models first
    print("Checking available Ollama models...")
    available_models = check_available_models()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for collection in os.listdir(base_dir):
        path = os.path.join(base_dir, collection)
        if os.path.isdir(path) and "Collection" in collection:
            print(f"Processing {collection}...")
            process_collection(path)