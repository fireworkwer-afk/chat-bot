# 🤖 Hybrid Local AI Assistant (RAG + Live Search)

An intelligent, privacy-focused hybrid AI chatbot created by **Abhi (coolbro_abhi)**. This chatbot works both **Online** (with live web search) and **Offline** (using local RAG with vector search and Ollama).

---

## ✨ Features

* 🌐 **Dual-Mode Operation:**
  * **Online Mode:** Automatically detects internet connectivity and fetches real-time news/information using DuckDuckGo/Google search.
  * **Offline Mode:** Seamlessly falls back to local knowledge base (`reviews.csv`) powered by ChromaDB vector store when internet is unavailable.
* 🔒 **100% Local LLM:** Runs fully on your hardware using **Ollama** (`qwen2.5:1.5b`) and local embeddings (`mxbai-embed-large`)[cite: 2].
* ⚡ **Retrieval-Augmented Generation (RAG):** Uses LangChain and ChromaDB to perform fast vector similarity search over your custom dataset[cite: 2].
* 🎨 **Interactive UI/UX:** Built with Streamlit for a clean, browser-based chat interface.

---

## 🛠️ Tech Stack

* **Language:** Python
* **LLM Runtime:** Ollama (`qwen2.5:1.5b`)[cite: 2]
* **Embedding Model:** `mxbai-embed-large`[cite: 2]
* **Framework:** LangChain[cite: 2]
* **Vector Database:** ChromaDB[cite: 2]
* **Web Search:** DuckDuckGo Search
* **UI Framework:** Streamlit

---

## 📁 Project Structure

```text
├── main.py              # Main application entry point (Streamlit UI + Internet Switching logic)
├── vector.py            # Script to parse CSV, embed documents, and setup ChromaDB vector store[cite: 2]
├── reviews.csv          # Local knowledge base dataset[cite: 2]
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation


1. Prerequisites
Make sure you have Ollama installed on your computer. Pull the required models via your terminal:

Bash
ollama pull qwen2.5:1.5b
ollama pull mxbai-embed-large

2. Clone the Repository
Bash
git clone [https://github.com/your-username/local-ai-assistant.git](https://github.com/your-username/local-ai-assistant.git)
cd local-ai-assistant
3. Set Up Virtual Environment (Optional but Recommended)
Bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Bash
pip install -r requirements.txt
🎮 How to Run
Initialize Vector Store (First Time Only):
Run vector.py to index your reviews.csv data into the local Chroma database[cite: 2].

Bash
python vector.py
Launch the Chatbot App:
Run the Streamlit app to open the chatbot in your web browser:

Bash
streamlit run main.py
