# PDF Language Extractor – Adobe Hackathon Round 1B

This tool processes a collection of PDFs and identifies the most relevant content based on a user **persona** and **job-to-be-done**. It uses **local language models** (via Ollama) and intelligent fallback strategies when models are unavailable.

---

## What the Code Does

This project consists of several key stages:

### 1. Input Parsing
- Reads the `challenge1b_input.json` file.
- Extracts the **persona** and **job-to-be-done** fields and combines them to form a **query**.

### 2. PDF Processing
- Opens each PDF file in the `PDFs/` directory using `PyMuPDF (fitz)`.
- For each page, extracts:
  - **Headings** (based on font size, formatting, etc.)
  - **Content** (full paragraph text below or around headings)

### 3. Embedding Text
- Attempts to use **Ollama** models to embed each section.
- If models are unavailable, it uses a **custom fallback embedding function** based on word frequencies, keywords, punctuation, and structure.

### 4. Section Ranking
- All section embeddings are compared with the query embedding using cosine similarity.
- A **diversity penalty** is applied to avoid ranking too many sections from the same document.
- Top `N` relevant sections are selected.

### 5. Summarization
- Uses **spaCy** to break down selected sections into readable summaries (first few important sentences).

### 6. Output Generation
- Writes results to a `challenge1b_output.json` file.
- Output contains:
  - Metadata
  - Top-ranked sections
  - Summaries of each relevant section

---

## Features

- PDF text extraction with heading recognition
- Embedding using multiple local models via Ollama
- Fallback embedding to ensure robustness
- Ranking logic with document diversity penalty
- Summarization of selected sections
- Fully offline processing

---

## Tech Stack

| Library     | Purpose                              |
|-------------|--------------------------------------|
| `fitz` (PyMuPDF) | Extract structured text from PDFs |
| `Ollama`    | Generate local text embeddings       |
| `spaCy`     | Sentence segmentation & summaries    |
| `NumPy`     | Embedding math and similarity        |
| `json`, `os`, `re`, `glob` | Utilities & I/O     |

---

## Folder Structure

Each collection folder should look like this:

```

CollectionName/
├── challenge1b\_input.json
├── challenge1b\_output.json  # (created after run)
├── PDFs/
│   ├── file1.pdf
│   ├── file2.pdf
│   └── ...

````

### Example `challenge1b_input.json`
```json
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "description": "Plan a trip of 4 days for a group of 10 college friends."
  }
}
````

---

## How to Run

### Step 1: Install Dependencies

```bash
pip install pymupdf spacy numpy ollama
```

Make sure you also download the Spacy English model:

```bash
python -m spacy download en_core_web_sm
```

### Step 2: Run the Processor

```bash
python process_collection.py
```

It automatically looks for all folders in the current directory with "Collection" in the name.

---

## 🔧 Supported Embedding Models (Ollama)

The system tries these models in order:

1. `nomic-embed-text`
2. `all-minilm`
3. `mxbai-embed-large`
4. `snowflake-arctic-embed`
5. `bge-large`


---

## Sample Output

Here’s a sample from the `challenge1b_output.json`:

```json
{
    "metadata": {
        "input_documents": [
            "South of France - Cities.pdf",
            "South of France - Cuisine.pdf",
            "South of France - History.pdf",
            "South of France - Restaurants and Hotels.pdf",
            "South of France - Things to Do.pdf",
            "South of France - Tips and Tricks.pdf",
            "South of France - Traditions and Culture.pdf"
        ],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
        "processing_timestamp": "2025-07-28T16:42:04.718452"
    },
    "extracted_sections": [
        {
            "document": "South of France - Tips and Tricks.pdf",
            "section_title": "The Ultimate South of France Travel Companion: Your Comprehensive Guide to Packing,",
            "importance_rank": 1,
            "page_number": 1
        },
        {
            "document": "South of France - Things to Do.pdf",
            "section_title": "Ultimate Guide to Activities and Things to Do in the South of France",
            "importance_rank": 2,
            "page_number": 1
        },
        {
            "document": "South of France - Cuisine.pdf",
            "section_title": "A Culinary Journey Through the South of France",
            "importance_rank": 3,
            "page_number": 1
        },
        {
            "document": "South of France - History.pdf",
            "section_title": "A Historical Journey Through the South of France",
            "importance_rank": 4,
            "page_number": 1
        },
        {
            "document": "South of France - Restaurants and Hotels.pdf",
            "section_title": "Comprehensive Guide to Restaurants and Hotels in the South of France",
            "importance_rank": 5,
            "page_number": 1
        }
    ],
    "subsection_analysis": [
        {
            "document": "South of France - Tips and Tricks.pdf",
            "refined_text": "Planning a trip to the South of France requires thoughtful preparation to ensure a comfortable and enjoyable experience. This guide covers everything from packing essentials to travel tips, catering to all seasons and various activities. Whether you're traveling solo, with kids, or in a group, this guide will help you make the most of your trip.",
            "page_number": 1
        },
        {
            "document": "South of France - Things to Do.pdf",
            "refined_text": "The South of France, with its stunning coastline, picturesque villages, and vibrant cities, oﬀers a wealth of activities and experiences for travelers. Whether you're seeking adventure, relaxation, or cultural enrichment, this region has something for everyone. This guide will take you through a variety of activities and must-see attractions to help you plan an unforgettable trip.",
            "page_number": 1
        },
        {
            "document": "South of France - Cuisine.pdf",
            "refined_text": "The South of France, known for its stunning landscapes and charming villages, is also a paradise for food lovers. The region's cuisine is a delightful blend of Mediterranean flavors, fresh ingredients, and traditional recipes passed down through generations. This guide will take you through the diﬀerent types of food, famous dishes, must-visit restaurants, renowned wine regions, and the types of wines they produce to help you plan an unforgettable culinary adventure.",
            "page_number": 1
        },
        {
            "document": "South of France - History.pdf",
            "refined_text": "The South of France, renowned for its picturesque landscapes, charming villages, and stunning coastline, is also steeped in history. From ancient Roman ruins to medieval fortresses and Renaissance architecture, this region oﬀers a fascinating glimpse into the past. This guide will take you through the histories of major cities, famous historical sites, and other points of interest to help you plan an enriching and unforgettable trip.",
            "page_number": 1
        },
        {
            "document": "South of France - Restaurants and Hotels.pdf",
            "refined_text": "The South of France, known for its stunning landscapes, rich cultural heritage, and exquisite cuisine, is a dream destination for travelers. From the glamorous beaches of the French Riviera to the charming villages of Provence, this region oﬀers a diverse array of experiences. Whether you're looking for budget-friendly options, family-friendly spots, upscale dining, or luxurious experiences, this guide will help you find the perfect restaurants and hotels for your trip.",
            "page_number": 1
        }
    ]
}
```

---

## Notes

* Embedding fallback is robust and keyword-aware.
* Only the top 5 most relevant sections are returned.
* Multiple documents are supported in one go.
* Summaries are capped at \~500 characters for readability.

---

##  Author

**Samyukta Gade**
**Sreeja Bommagani**
