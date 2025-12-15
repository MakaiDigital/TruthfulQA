# AI Data Engineering Challenge: The Truthful RAG

Welcome! This exercise is designed to test your ability to build data pipelines for AI applications, handle vector storage, and implement a basic RAG (Retrieval-Augmented Generation) system.

## 🛑 Repository Instructions

Before writing any code, please follow these steps to set up your workspace:

1.  **Clone this repository.**
2.  **Create a new branch** using the naming convention `candidate/firstname-lastname`.
    ```bash
    git checkout -b candidate/jane-doe
    ```
3.  **Commit your work** to this branch.
4.  When you are finished, **push your branch** and open a Pull Request against `main`.

---

## 🕒 Timeline
We expect this task to take approximately **3 to 4 hours of hands-on coding**, though you have a window of **48 hours** from receiving the link to submit your solution.

---

## 🧠 The Context
We are building a chatbot designed to answer questions truthfully and avoid common misconceptions. To do this, we need to build a Knowledge Base derived from verified data.

We have provided the **TruthfulQA** dataset in the `data/` folder of this repository. This dataset specifically pairs questions with both "correct/truthful" answers and "incorrect/misconception" answers.

### The Goal
Your objective is to build a data pipeline that ingests this local dataset, filters out the lies, indexes the truths into a Vector Database, and exposes a simple interface to query that data.

---

## 🛠 The Assignment

### Part 1: The Data Pipeline (ETL & Vectorization)
Create a Python script/module that performs the following:

1.  **Ingestion:** Load the dataset from the file(s) located in the `data/` directory.
2.  **Cleaning & Processing:** * We want our bot to learn *only* the truth. Filter the data to extract the text from `Best Answer` and `Correct Answers`.
    * **Important:** Ensure that text from the `Incorrect Answers` column is **completely excluded** from the knowledge base.
    * *Optional:* extract source URLs as metadata.
3.  **Embedding & Storage:**
    * Generate embeddings for the truthful facts using an open-source model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) or an API of your choice.
    * Store the vectors and metadata in a local Vector Database (e.g., ChromaDB, FAISS, Qdrant, or basic LanceDB).

### Part 2: The RAG System
Implement a simple class or script `rag_system.py` that handles the following:

1.  **Retrieval:** A function `retrieve_context(query)` that searches your Vector DB for the most relevant facts given a user question.
2.  **Generation:** A function `generate_answer(query, context)` that synthesizes an answer.
    * *Note:* You may use a free API (e.g., HuggingFace Inference), a local LLM (e.g., Ollama), or OpenAI. 
    * *Alternative:* If you do not have access to an LLM, a "mock" generator that formats and returns the top 3 retrieved documents is acceptable, provided the retrieval logic is sound.

### Part 3: Evaluation
Create a script `evaluate.py` that:

1.  Selects **20 random questions** from the dataset.
2.  Runs them through your RAG pipeline.
3.  Calculates the **Cosine Similarity** between your generated answer and the dataset's `Best Answer` (using the same embedding model used for storage).
4.  Outputs the results to a CSV file named `results.csv` with columns: `Question | Generated Answer | Similarity Score`.

---

## 📦 Deliverables

1.  **Code:** Clean, modular Python code.
2.  **Infrastructure:** A `Dockerfile` (and optional `docker-compose.yml`) that sets up the environment. We should be able to run your pipeline with one or two commands.
3.  **Requirements:** A `requirements.txt` or `pyproject.toml` file.
4.  **Documentation:** Update the `README.md` (or add a `SOLUTION.md`) with:
    * Instructions on how to run the ingestion and evaluation.
    * A brief explanation of your design choices (Which embedding model? Which vector store? How did you chunk the text?).
    * Any challenges you faced.

---

## ⚖️ Evaluation Criteria

We are looking for:
* **Data Integrity:** Did you successfully separate the "Correct" answers from the "Incorrect" ones?
* **Engineering Practices:** Is the code readable, modular, and easy to run via Docker?
* **Architecture:** Does the vector search implementation make sense?
* **Completeness:** Does the evaluation script run and produce the CSV?

Good luck! We look forward to seeing your solution.