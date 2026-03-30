# Philippine Fact Checker 🧠🔎
An AI-powered fact-checking chatbot built with Python, Streamlit, and LLMs (Llama 3.1 via Groq API), designed to verify claims using real-time web data, semantic similarity search, and adaptive learning.

🚀 Features
Asynchronous Data Ingestion: Fetches real-time information using the Google Custom Search API and httpx for async HTTP requests.

Semantic Search Matching: Uses SentenceTransformer embeddings (all-MiniLM-L6-v2) to retrieve previously verified high-confidence responses.

Feedback-Driven Learning: Stores user feedback (good/bad) into a SQLite database to enable continuous system improvement.

Data Caching: Optimizes repeated search queries with Streamlit's built-in caching.

Structured Parsing: Extracts, transforms, and displays relevant snippets, verdicts, explanations, and source links.

Full-Stack Application: Integrated frontend (Streamlit chat UI) and backend (data processing pipelines).

🛠️ Tech Stack
Languages: Python

Frameworks: Streamlit

APIs: Google Custom Search API, Groq LLM API

Database: SQLite

Libraries: SentenceTransformers, httpx, asyncio, dotenv

📈 Key Highlights
Designed an efficient and scalable data pipeline for claim verification tasks.

Implemented lightweight MLOps practices for feedback-based quality improvement.

Focused on data engineering principles: ingestion → transformation → storage → retrieval.

📂 Future Improvements
Fine-tune retrieval thresholds for semantic search.

Expand feedback types (e.g., detailed tagging).

Integrate image search for misinformation detection.
