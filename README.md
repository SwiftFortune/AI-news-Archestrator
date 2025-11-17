# 📰 News Orchestrator — AI-Powered Multi-Source News Analyzer

The **News Orchestrator** is an AI-driven Streamlit dashboard that fetches news from multiple sources (via Google News RSS), summarizes events using NLP, extracts chronological timelines, evaluates source reliability, and presents everything in a clean, interactive UI.

This project is perfect for:
- Journalists  
- Students  
- Researchers  
- Analysts  
- Anyone looking for fact-based, concise, multi-source news insights  

---

## 🚀 Features

### ✅ 1. **AI Progressive Summary**
- Generates short, fact-driven summaries using **DistilBART CNN summarization model**  
- Summarizes multiple articles and merges them into one final output  
- Intelligent fallback for short or incomplete content  

---

### ✅ 2. **Automated Timeline Extraction**
- Uses **spaCy NER** to identify dates and events  
- Creates a **chronological event progression**  
- Displays a clean vertical timeline  

---

### ✅ 3. **News Fetching (RSS Based)**
- Fetches top articles from **Google News RSS** for any topic  
- Cleans noisy HTML content  
- Extracts title, description, link, and source  

---

### ✅ 4. **Source Reliability Score**
- Computes an approximate reliability score based on:
  - Content length  
  - Depth of coverage  
- Visualized using a modern **Plotly Gauge Chart**

---

### ✅ 5. **Clean, Modern UI**
- Beautiful Streamlit theme with cards, metrics, and responsive layout  
- Two-column summary + timeline section  
- Right-side metrics panel  
- Smooth scrolling timeline  

---

## 📂 Project Structure

```
News-Orchestrator/
│
├── app.py           # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md        # Documentation (this file)
└── assets/          # Optional images, screenshots, icons
```

---

## 🔧 Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/news-orchestrator.git
cd news-orchestrator
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download spaCy Model**
```bash
python -m spacy download en_core_web_md
```

or fallback:

```bash
python -m spacy download en_core_web_sm
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser:

```
http://localhost:8501
```

---

## 🧠 Technologies Used

| Technology | Purpose |
|-----------|----------|
| **Streamlit** | UI Dashboard |
| **Google News RSS** | Collecting News Articles |
| **spaCy NLP** | NER for dates/events |
| **HuggingFace Transformers** | Summarization (DistilBART) |
| **Plotly** | Gauge visualizations |
| **BeautifulSoup** | HTML cleaning |
| **dateparser** | Date normalization |

---

## 📸 Screenshots (Add Your Own)

| Feature | Screenshot |
|--------|------------|
| Summary Section | ![Summary Screenshot](assets/summary.png) |
| Timeline | ![Timeline Screenshot](assets/timeline.png) |
| Reliability Gauge | ![Gauge Screenshot](assets/gauge.png) |

(Replace with real images)

---

## 🛠 Code Overview

### 🧼 Cleaning & Preprocessing
- HTML Removal  
- Short-text fallback  
- Combined text logic  

### 🧠 Summarization Engine
- Uses `sshleifer/distilbart-cnn-12-6`  
- Safe fallback on summarization errors  

### ⏳ Timeline Generator
- spaCy NER for:
  - DATE  
  - EVENT  
- Date normalization using dateparser  

### 📊 Reliability Scoring
- Based on average word count of articles  
- Normalized 0–1  

---

## 📝 Future Enhancements

- Sentiment analysis per article  
- Stance detection (support, neutral, against)  
- Duplicate article merging  
- Deep narrative generation  
- Topic clustering  
- More advanced reliability scoring  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open Issues or submit Pull Requests.




