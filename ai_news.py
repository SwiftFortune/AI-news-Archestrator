import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import spacy
from transformers import pipeline
import numpy as np
import plotly.graph_objects as go
import dateparser 

st.set_page_config(page_title="News Orchestrator", layout="wide", page_icon="📰")

# -----------------------------------------------------------
# Custom CSS for modern dashboard design
# -----------------------------------------------------------
st.markdown("""
<style>

html, body, .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin: 0 !important;
}

.block-container {
    max-width: 96%;
}

.header-bar {
    width: 100%;
    padding: 10px 25px;
    background-color: #1f2937;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 6px;
    margin-bottom: 10px;
}

.header-title {
    font-size: 20px;
    font-weight: 600;
}

.card {
    background-color: white;
    padding: 16px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}

.summary-box {
    padding: 14px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.5;
    max-height: 350px; 
    overflow-y: auto;
    background: #f1f5f9;
    border-left: 4px solid #3b82f6;
}

.metric {
    font-size: 15px;
    margin-bottom: 10px;
}

.timeline-event {
    border-left: 3px solid #3b82f6;
    padding-left: 15px;
    margin-bottom: 15px;
    position: relative;
}

.timeline-event::before {
    content: '•';
    color: #3b82f6;
    font-size: 1.5em;
    position: absolute;
    left: -11px;
    top: 0px;
    background-color: white;
    border-radius: 50%;
    line-height: 0.7;
}


.timeline-date {
    font-weight: bold;
    color: #1e40af; 
    font-size: 14px;
    margin-bottom: 2px;
}

h2 {
    margin-bottom: 6px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Load ML Models (Cached for performance)
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads and caches the spaCy NLP model and the summarization pipeline."""
    try:
        nlp = spacy.load("en_core_web_md")
    except:
        nlp = spacy.load("en_core_web_sm") 
    
    # Load Hugging Face Summarization Pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return nlp, summarizer

nlp, summarizer = load_models()

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def fetch_news_rss(topic, max_articles=15):
    """Fetches articles from Google News RSS feed."""
    url = f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:max_articles]
        articles = []
        for item in items:
            title = item.title.text if item.title else ""
            desc = item.description.text if item.description else ""
            link = item.link.text if item.link else "#" 
            source = item.source.text if item.source else "Unknown"
            articles.append({"title": title, "content": desc, "source": source, "link": link})
        return pd.DataFrame(articles)
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return pd.DataFrame(columns=["title", "content", "source", "link"])


def clean_text(t):
    """Removes HTML and normalizes whitespace."""
    t = re.sub(r"<.*?>", "", str(t))
    return re.sub(r"\s+", " ", t).strip()


def summarize_articles(texts):
    """
    Generates a progressive, fact-based summary. 
    NOTE: The text filtering is now handled *before* calling this function.
    """
    if not texts:
        # Fallback should theoretically never be hit now
        return "No text available for summary (unexpected error)."

    individual_summaries = []
    for text in texts:
        try:
            # Skip summarization if input is literally just a title (very short)
            if len(text.split()) < 5: 
                individual_summaries.append(text)
                continue
            
            summary_piece = summarizer(
                text, 
                max_length=100, 
                min_length=30, 
                do_sample=False
            )[0]["summary_text"]
            individual_summaries.append(summary_piece)
        except:
            # If summarization fails (e.g., token error), use the input text as a fallback
            individual_summaries.append(text) 

    final_summary = " ".join(individual_summaries)
    
    # Ensure the final output is not empty
    if not final_summary.strip():
        return "Analysis yielded no useful summary."
        
    return final_summary[:1000] + "..." if len(final_summary) > 1000 else final_summary


def generate_timeline(df):
    """Uses spaCy NER to extract chronological milestones from articles."""
    timeline_events = []
    
    # Ensure the combined_text column exists (created in the main block)
    if 'combined_text' not in df.columns:
        return []
    
    for index, row in df.iterrows():
        # Use the combined text for NER
        doc = nlp(row['combined_text'])
        article_title = row['title'] 
        
        for ent in doc.ents:
            if ent.label_ == "DATE" or ent.label_ == "EVENT":
                
                sentence = ent.sent.text.strip()
                
                if len(sentence.split()) < 5: 
                    continue 

                milestone = sentence.replace(article_title, "").strip()
                
                if not milestone:
                    continue

                # Corrected dateparser setting
                parsed_date = dateparser.parse(ent.text, settings={'RELATIVE_BASE': pd.to_datetime('today'), 'PREFER_DATES_FROM': 'past'})
                
                if parsed_date:
                    date_str = parsed_date.strftime("%Y-%m-%d")
                    
                    timeline_events.append({
                        "date": parsed_date,
                        "date_str": date_str,
                        "milestone": milestone.capitalize(), 
                        "source": row['source'],
                        "link": row['link']
                    })

    # Remove duplicates and sort chronologically
    unique_events = {}
    for event in timeline_events:
        key = (event['date_str'], event['milestone'])
        if key not in unique_events:
            unique_events[key] = event
    
    sorted_timeline = sorted(unique_events.values(), key=lambda x: (x['date'], x['source']))
    
    return sorted_timeline


def score_source(source, df):
    """Calculates a rudimentary score based on content depth."""
    arts = df[df["source"] == source]
    if arts.empty:
        return 0
    length = np.mean(arts["content"].apply(lambda x: len(str(x).split())))
    return min(1.0, length / 200)


def create_gauge(score):
    """Creates a Plotly Gauge chart for reliability score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#10b981"}, 
            "steps": [
                {"range": [0, 40], "color": "#f87171"}, 
                {"range": [40, 70], "color": "#facc15"}, 
                {"range": [70, 100], "color": "#d9f99d"}, 
            ]
        }
    ))
    fig.update_layout(height=210, margin=dict(t=0, b=0, l=0, r=0))
    return fig


# -----------------------------------------------------------
# Header Bar
# -----------------------------------------------------------
st.markdown("""
<div class="header-bar">
    <div class="header-title">📰 News Orchestrator</div>
    <div class="header-user">Welcome, Analyst</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Main Layout
# -----------------------------------------------------------
# Changed column layout from [5, 1] to a single column for the summary, or 
# if you want the summary and timeline side-by-side, we'll revert to 2 columns for the main content.
# Let's keep the side-by-side display for better use of wide screen space.
left_main, right_metrics = st.columns([5, 1])

with st.sidebar:
    st.markdown("## Event Input")
    topic = st.text_input("Enter Topic/Event Title:", "Russia Ukraine War")
    
    generate = st.button("🚀 Generate Dashboard", use_container_width=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
    st.session_state.timeline = []
    st.session_state.summary = "Enter a topic and click 'Generate Dashboard' to start analysis."
    st.session_state.avg_score = 0.0

if generate:
    with st.spinner(f"Fetching and analyzing news for **{topic}**..."):
        df = fetch_news_rss(topic)
        st.session_state.df = df
        
        if not df.empty:
            df["clean"] = df["content"].apply(clean_text)
            
            # --- CRITICAL MODIFICATION: Use title as fallback if content is too short ---
            def get_combined_text(row):
                # If cleaned content is less than 5 words, use the title instead
                if len(row['clean'].split()) < 5:
                    return row['title']
                return row['clean']
                
            df['combined_text'] = df.apply(get_combined_text, axis=1)
            combined_texts = df['combined_text'].tolist()
            # -------------------------------------------------------------------------
            
            st.session_state.summary = summarize_articles(combined_texts)
            # st.session_state.narrative is removed here
            st.session_state.timeline = generate_timeline(df)
            
            df["reliability"] = df["source"].apply(lambda s: score_source(s, df))
            st.session_state.avg_score = float(df["reliability"].mean())
        else:
            st.warning("Could not fetch articles for this topic. Check your internet or try a different topic.")
            st.session_state.summary = "No news articles found."
            st.session_state.timeline = []
            st.session_state.avg_score = 0.0


# --- Display Results ---

with left_main:
    st.markdown(f"## {topic}")
    
    # Use two columns for the Summary and Timeline side-by-side
    col_summary, col_timeline = st.columns(2)

    with col_summary:
        # 1. PROGRESSIVE SUMMARY CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Progressive Summary (Key Facts)")
        st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_timeline:
        # 3. TIMELINE CARD (moved to be next to summary)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### ⏳ Event Timeline: Systematic Progression")
        
        if st.session_state.timeline:
            st.markdown(f"**Total Milestones Extracted:** {len(st.session_state.timeline)}")
            
            # Use a scrollable container for the timeline
            with st.container(height=280): 
                for event in st.session_state.timeline:
                    milestone_text = event['milestone']
                    
                    st.markdown(f"""
                    <div class='timeline-event'>
                        <div class='timeline-date'>🗓️ {event['date_str']}</div>
                        **Milestone:** {milestone_text}
                        <p style='font-size:12px; margin-top:5px; color:#6b7280;'>Source: 
                            <a href='{event['link']}' target='_blank' style='color:#3b82f6;'>{event['source']}</a>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        elif st.session_state.df.empty and not generate:
            st.info("Click 'Generate Dashboard' in the sidebar to view the chronological timeline.")
        else:
            st.info("AI could not extract specific chronological milestones. Try a topic with clearer dates/events and sufficient content.")
        
        st.markdown("</div>", unsafe_allow_html=True)


with right_metrics:
    if not st.session_state.df.empty or generate:
        articles = len(st.session_state.df)
        sources = st.session_state.df["source"].nunique()

        # 4. METRICS CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Key Metrics")
        st.markdown(f"<div class='metric'>📝 <b>{articles}</b> Articles Fetched</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>📌 <b>{len(st.session_state.timeline)}</b> Milestones Extracted</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>📰 <b>{sources}</b> Unique Sources</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. RELIABILITY CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Source Reliability Score (Content Depth)")
        st.plotly_chart(create_gauge(st.session_state.avg_score), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)