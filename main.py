import streamlit as st 
import requests
import sqlite3
import numpy as np
from datetime import datetime
from groq import Groq
from sentence_transformers import SentenceTransformer
import sys
import asyncio
from dotenv import load_dotenv
import os
import httpx  # For async HTTP calls

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set API keys and configuration
# Access API keys from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CX_ID = st.secrets["GOOGLE_CX_ID"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# Initialize sentence transformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Database Functions for Feedback Storage ---

def init_feedback_db():
    """Initialize the feedback database and create the feedback table if it does not exist."""
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_index INTEGER,
            message_content TEXT,
            feedback_type TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_feedback(message_index, message_content, feedback_type):
    """Store a feedback record in the database."""
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO feedback (message_index, message_content, feedback_type, timestamp)
        VALUES (?, ?, ?, ?)
    """, (message_index, message_content, feedback_type, timestamp))
    conn.commit()
    conn.close()

# Initialize the feedback database on startup
init_feedback_db()

# --- Functions for Feedback Retrieval for Adaptive Learning ---

def fetch_good_feedback():
    """Fetch queries and responses with 'good' feedback from the feedback database."""
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("SELECT message_index, message_content FROM feedback WHERE feedback_type = 'good'")
    records = c.fetchall()
    conn.close()
    return records  # List of tuples: (message_index, message_content)

def get_similar_good_response(new_query, threshold=0.8):
    """Find a good response from stored feedback similar to the new query."""
    records = fetch_good_feedback()
    if not records:
        return None

    new_query_emb = embedder.encode(new_query, convert_to_tensor=True)
    best_score = -1
    best_response = None
    for idx, content in records:
        # In production, store the original query alongside the response.
        stored_emb = embedder.encode(content, convert_to_tensor=True)
        cosine_sim = np.dot(new_query_emb, stored_emb) / (np.linalg.norm(new_query_emb) * np.linalg.norm(stored_emb))
        if cosine_sim > best_score and cosine_sim >= threshold:
            best_score = cosine_sim
            best_response = content
    return best_response

# --- Asynchronous Function for Fact Checking using Google API ---

async def perform_google_search(query):
    """Asynchronously perform a Google Custom Search API call using httpx."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX_ID,
        "q": query,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

# --- Caching the Google Search results ---
# This synchronous wrapper uses caching. It calls the async function via asyncio.run.
@st.cache_data(show_spinner=True)
def cached_google_search(query):
    return asyncio.run(perform_google_search(query))

def extract_search_results(search_json):
    """Extract snippets and links from the Google search results."""
    results = []
    if 'items' in search_json:
        for item in search_json['items']:
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            results.append(f"Snippet: {snippet}\nLink: {link}")
    return "\n\n".join(results)

def generate_llm_response(prompt):
    """Generate a response from the Groq LLM using the provided prompt."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    return response.choices[0].message.content.strip()

def fact_check(query):
    """
    Perform search, build prompt, and generate fact-check response.
    The LLM is expected to output the answer in the following format:
    
    Reasoning: <detailed internal reasoning here (hidden in UI)>
    
    Final Verdict: <your final verdict here (True, False, Misleading, or Lack of Evidence)>
    
    Explanation: <a brief explanation of your verdict (shown by default)>
    
    Sources: <relevant source links if applicable (hidden in UI)>
    """
    # Use the cached (and thus pre-fetched if repeated) async Google search call:
    search_json = cached_google_search(query)
    search_context = extract_search_results(search_json)
    
    # Check for similar good feedback to use as additional context
    similar_good = get_similar_good_response(query)
    context_note = f"Past Verified Response:\n{similar_good}\n\n" if similar_good else ""
    
    prompt = (
        f"Query: {query}\n\n"
        f"{context_note}"
        f"Search Results:\n{search_context}\n\n"
        "You are a professional fact checker. Your task is to analyze the above information and determine a verdict about the claim based on the available evidence. "
        "However, do not rely solely on the provided search results if they are incomplete. If the results do not fully capture the context or contain conflicting information, perform an independent analysis to verify the claim. "
        "Provide a well-structured response. Please follow the exact format below:\n\n"
        "Reasoning: <provide a brief summary of your reasoning including any analysis. This section will be hidden in the UI>\n\n"
        "Final Verdict: <If the evidence clearly supports or refutes the claim, state one of the following verdicts: True, False, Misleading, or Lack of Evidence. If the evidence is inconclusive, simply skip the verdict step.>\n\n"
        "Explanation: <a brief explanation of your verdict that will be displayed normally>\n\n"
        "Sources: <relevant source links, if applicable. This section will be hidden in the UI>\n\n"
        "Please ensure that your response is clear and concise, and avoid unnecessary verbosity."
    )
    answer = generate_llm_response(prompt)
    return answer

def parse_response(response):
    """
    Splits the LLM response into four parts based on the following markers:
    
    Reasoning: <hidden internal reasoning>
    Final Verdict: <final verdict>
    Explanation: <display explanation>
    Sources: <hidden source links>
    """
    # Define markers
    reasoning_marker = "Reasoning:"
    verdict_marker = "Final Verdict:"
    explanation_marker = "Explanation:"
    sources_marker = "Sources:"
    
    # Initialize parts
    reasoning = ""
    verdict = ""
    explanation = ""
    sources = ""
    
    # Find positions
    pos_reasoning = response.find(reasoning_marker)
    pos_verdict = response.find(verdict_marker)
    pos_explanation = response.find(explanation_marker)
    pos_sources = response.find(sources_marker)
    
    if pos_reasoning != -1 and pos_verdict != -1:
        reasoning = response[pos_reasoning+len(reasoning_marker):pos_verdict].strip()
    if pos_verdict != -1 and pos_explanation != -1:
        verdict = response[pos_verdict+len(verdict_marker):pos_explanation].strip()
    if pos_explanation != -1:
        # Explanation is assumed to go until Sources if present, else until end of response.
        if pos_sources != -1:
            explanation = response[pos_explanation+len(explanation_marker):pos_sources].strip()
        else:
            explanation = response[pos_explanation+len(explanation_marker):].strip()
    if pos_sources != -1:
        sources = response[pos_sources+len(sources_marker):].strip()
    
    return reasoning, verdict, explanation, sources

# --- Streamlit Chat UI Setup ---

st.set_page_config(
    page_title="Philippine Fact Checker",
    page_icon=":mag:",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>Philippine Fact Checker</h1>", unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # each element: {"role": "user"/"assistant", "content": str}

# Display chat history in a chat-like interface
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            # Parse the assistant message into its sections
            reasoning_details, final_verdict, explanation_text, sources_links = parse_response(msg["content"])
            
            # First: Reasoning (hidden within a collapsible expander)
            if reasoning_details:
                with st.expander("Show detailed reasoning"):
                    st.write(reasoning_details)
            
            # Second: Final Verdict (displayed by default)
            st.markdown(f"**Final Verdict**: {final_verdict}")
            
            # Third: Explanation (displayed by default)
            st.markdown(f"**Explanation**: {explanation_text}")
            
            # Fourth: Sources (hidden within a collapsible expander)
            if sources_links:
                with st.expander("Show relevant sources"):
                    st.write(sources_links) 

# Input area for new user message
user_input = st.chat_input("Enter a claim to fact-check........")

if user_input:
    # Display the user input in the chat UI
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Run fact checking with a spinner
    with st.spinner("Fact checking..."):
        try:
            # Note: The Google search is done asynchronously and its result is cached
            answer = fact_check(user_input)
        except Exception as e:
            answer = f"An error occurred: {e}"
    
    # Append the assistant's answer to the chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    new_idx = len(st.session_state.chat_history) - 1
    
    with st.chat_message("assistant"):
        # Parse the answer for the four sections
        reasoning_details, final_verdict, explanation_text, sources_links = parse_response(answer)
        
        # Reasoning (hidden within an expander)
        if reasoning_details:
            with st.expander("Show detailed reasoning"):
                st.write("**Reasoning**:")
                st.write(reasoning_details)
        
        # Final Verdict and Explanation are shown directly
        st.markdown(f"**Final Verdict**: {final_verdict}")
        st.markdown(f"**Explanation**: {explanation_text}")
        
        # Sources (hidden within an expander)
        if sources_links:
            with st.expander("Show relevant sources"):
                st.write("**Sources**:")
                st.write(sources_links)
                
        # Feedback buttons for the new assistant message
        col1, col2, col3 = st.columns([1, 1, 10]) 
        with col1:
            if st.button("👍", key=f"good_{new_idx}"):
                store_feedback(new_idx, answer, "good")
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎", key=f"bad_{new_idx}"):
                store_feedback(new_idx, answer, "bad")
                st.error("Thank you for your feedback! We'll strive to improve.")
