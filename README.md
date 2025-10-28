
# 🤖 Docubot

**Docubot** is an AI-powered intelligent document assistant that allows users to **upload, search, summarize, and chat** with their documents and web content.  
It uses **embeddings**, **vector databases**, and **LLMs** to provide accurate, context-aware answers — all through a modern **Streamlit interface** with **user authentication** and **MongoDB persistence**.

---

## 🧠 What is Docubot?

Docubot is designed to help users quickly extract insights from **large document collections** or **web pages**.  
Instead of manually searching, you can simply ask Docubot questions like:

> “Summarize this PDF.”  
> “What are the key points from this medical paper?”  
> “What does this webpage say about data privacy?”

It’s like having your own **AI research assistant** — capable of understanding, retrieving, and summarizing complex text.

---

## 🚀 Key Features

✅ **Multi-format Document Ingestion**
- Supports **PDF, DOCX, TXT, and Markdown** files  
- Built-in **web scraping** to fetch and process content from URLs

✅ **Smart Query Processing**
- Uses vector embeddings to **find the most relevant information**
- Combines retrieval and generation for **accurate, grounded responses**

✅ **User Authentication**
- Secure **login and signup system** via MongoDB  
- Tracks user activity, uploads, and queries

✅ **Persistent Data Storage**
- All uploaded files, web scrapes, and query logs are stored in **MongoDB**

✅ **Streamlit Web Interface**
- Simple, interactive web UI  
- Upload documents, scrape websites, and chat with your data

✅ **Analytics & Logs**
- Each query, upload, and scrape is logged with timestamps and metrics

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend/UI** | Streamlit |
| **Backend** | Python |
| **Database** | MongoDB |
| **Vector Store** | FAISS / Qdrant |
| **Embeddings & LLMs** | OpenAI / Groq |
| **Auth & User Management** | MongoDB Users Collection |
| **Web Scraping** | Requests + BeautifulSoup |

---

## 📂 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Streamlit main app entry point |
| `auth.py` | Handles user authentication (login/signup) |
| `config.py` | Environment configuration and API key management |
| `data_processing.py` | Parses and preprocesses uploaded documents |
| `database.py` | MongoDB connection and CRUD operations |
| `query_processor.py` | Manages query understanding and LLM responses |
| `vector_store.py` | Handles vector embedding storage/retrieval |
| `web_scraper.py` | Fetches and processes website content |
| `requirements.txt` | Python dependencies list |

---

## ⚙️ Configuration

Before running Docubot, create a `.env` file in your project root with:

```env
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
MONGODB_URI=your-mongodb-uri
````

| Variable         | Description                                      |
| ---------------- | ------------------------------------------------ |
| `OPENAI_API_KEY` | For generating embeddings and LLM responses      |
| `QDRANT_URL`     | Vector database endpoint (if using Qdrant Cloud) |
| `QDRANT_API_KEY` | Auth key for Qdrant                              |
| `GROQ_API_KEY`   | Alternative LLM provider (optional)              |
| `MONGODB_URI`    | MongoDB connection string for user data & logs   |

> 💡 Tip: Add `.env` to `.gitignore` to keep keys private.

---

## 🧾 Quickstart

### 1️⃣ Clone and Setup

```bash
git clone https://github.com/maheshLEO4/Docubot.git
cd Docubot
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run app.py
```

> Open the link shown in your terminal (usually `http://localhost:8501`).

---

## 🧠 How It Works

1. **User Authentication:** Login or register securely via MongoDB.
2. **Upload Documents / Scrape Websites:** Files and web content are parsed and chunked.
3. **Embedding Creation:** Text chunks are converted into vector embeddings.
4. **Storage:** Embeddings are saved in Qdrant (or FAISS) for quick retrieval.
5. **Query:** When a question is asked, the most relevant chunks are fetched.
6. **LLM Response:** The LLM (OpenAI or Groq) generates a contextual answer.
7. **Logging:** User queries, responses, and timestamps are stored in MongoDB.

---

## 🧑‍💻 Example Usage

### Upload and Query a Document

1. Login to Docubot.
2. Upload your PDF, DOCX, or text file.
3. Ask questions like:

   * “Summarize this document.”
   * “What is the conclusion section about?”
   * “List all key findings.”

### Web Scraping

1. Enter a website URL in the app.
2. Docubot fetches and extracts main content.
3. Ask, “What does this article talk about?” or “Summarize this site.”

---

## 🧰 MongoDB Integration

Each user’s activity is securely logged:

| Collection     | Purpose                                |
| -------------- | -------------------------------------- |
| `users`        | Stores user credentials & session data |
| `file_uploads` | Tracks uploaded documents              |
| `web_scrapes`  | Records scraped website content        |
| `query_logs`   | Logs all user queries and responses    |

---

## 🧱 Authentication System

* **Sign Up / Login:** Users can register with email & password.
* **Session Handling:** Tracks last login and user activity.
* **Data Isolation:** Each user’s uploads, scrapes, and logs are private.

---

## 📦 Requirements

* Python 3.9+
* MongoDB (local or cloud)
* Streamlit
* OpenAI / Groq API access
* Qdrant or FAISS vector database

---

## 🧑‍💼 Why It’s Useful

* Saves time by **automating document comprehension**
* Useful for **students, researchers, and professionals**
* Enables **knowledge extraction** from massive text data
* Allows **secure, personalized** access and history tracking

---

## 📜 License

This project is open source under the **MIT License**.

---

## 📬 Contact

**Maintainer:** [@maheshLEO4](https://github.com/maheshLEO4)
**Project:** [Docubot GitHub Repository](https://github.com/maheshLEO4/Docubot)

---

