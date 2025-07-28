# 📄 Adobe Hackathon - PDF Processor

This project is a high-performance, offline Python-based PDF processing tool built for the Adobe Hackathon. It extracts structured outlines (headings), infers titles, and detects the primary language of each PDF file using smart heuristics — all optimized for accuracy and memory efficiency.

---

## 🚀 Features

- 📚 **Heading Detection:**  
  Extracts document structure (`H1`–`H4`) based on visual features like font size, boldness, alignment, and text content.

- 🏷️ **Title Inference:**  
  Smartly identifies the document title using layout and typography cues on the first page.

- 🌐 **Language Detection:**  
  Detects the primary language of the document using `langdetect`, with confidence scoring and fallback support.

- 🧹 **Noise Filtering:**  
  Ignores repetitive, irrelevant, and decorative text elements (e.g., footers, emails, links).

- 💾 **Memory Efficient:**  
  Chunk-based processing with garbage collection ensures smooth handling of large and complex PDFs.

---

## 🛠️ Setup Instructions

### 1. 🔧 Install Requirements
```bash
pip install PyMuPDF langdetect
```

### 2. 📂 Project Structure
```
/app
├── input/     # Place your PDF files here
├── output/    # JSON output files will be generated here
├── process_collection.py
```

### 3. ▶️ Run the Script
```bash
python process_collection.py
```

Each processed PDF will generate a `.json` file in the `output/` folder with the following format:

```json
{
  "title": "Sample Document Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Methodology", "page": 3 }
  ],
  "language": {
    "primary_language": "en",
    "language_name": "English",
    "confidence": 0.987,
    "detected_languages": [...]
  }
}
```

---

## 🧪 Example Use Cases

- Creating document previews or summaries
- Auto-generating Table of Contents
- Language-based filtering or indexing
- Preprocessing PDFs for AI/ML tasks

---

## ⚠️ Known Limitations

- Works best with machine-generated PDFs (not scanned images)
- Detection may be limited in extremely short or mixed-language docs
- It doesn't extract paragraph-level body content — only structure

---

## 📄 License

MIT License © 2025 Samyukta Gade, Sreeja Bommagani

---

## 🙌 Acknowledgements

Built with ❤️ for the Adobe Hackathon Round 1B.
