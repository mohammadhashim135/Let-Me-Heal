# **LetMeHeal** 🌐
**Your Secure, Interactive, and AI-Powered Medical Assistant**  

## **Overview**  
**LetMeHeal** is a Streamlit-based AI chatbot designed to provide reliable medical insights. Powered by HuggingFace LLMs, FAISS vector database, and Langchain, it empowers users to explore healthcare topics safely and accurately. LetMeHeal retrieves answers from verified medical literature, ensuring trust and precision in its responses.

---
Features 🚀

✅ **Medical Q&A Chatbot** – Ask medical questions and receive contextual, reliable answers.

✅ **Streamlit UI** – Interactive, user-friendly interface with persistent chat history.

✅ **RAG Pipeline** – Retrieval Augmented Generation for precise, context-based answers.

✅ **FAISS Integration** – Fast similarity search over embedded medical knowledge.

✅ **HuggingFace LLM** – Uses powerful language models for accurate responses.

✅ **Modular & Scalable** – Designed for continuous improvement and easy customization.

✅ **Secure Tokens** – Environment-managed API tokens using dotenv.

---

## **Tech Stack** 🛠  
- **Frontend/UI:** Streamlit

- **Backend:** Python, Langchain, HuggingFace

- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

- **Vector Database:** FAISS

- **Document Loading:** Langchain’s PyPDFLoader, DirectoryLoader

- **Prompt Engineering:** Custom PromptTemplate for medical Q&A

- **Environment Management:** dotenv (.env files)
---

## **Installation & Setup** 🏗  

### **1. Clone the Repository**  
```bash
git clone https://github.com/mohammadhashim135/Let-Me-Heal
cd Let-Me-Heal
```

### **2. Create a Virtual Environment**
```bash
python -m venv .venv

# Activate it:

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```


### **4. Start the Application**
```bash
streamlit run medical_chatbot.py
```
---

## **Usage Guide** 📝

🔹 Ask any medical question in the input box.

🔹 Receive contextual, reliable answers powered by AI and verified medical literature.

🔹 Chat history is saved in your session for continuous interaction.

🔹 Use it for learning, quick clarification, and preliminary health awareness.

---

## **Project Structure** 📂
```bash
LetMeHeal/
│
├── venv/                     # Python virtual environment
├── env/                      # Environment variables folder
├── requirements.txt          # Python package dependencies
│
├── medical_chatbot.py        # Main Streamlit app entry point
├── connect_with_llm.py       # LLM integration logic
├── memory_llm.py             # Memory management for LLM
│
├── vectorstore/
│   ├── index.faiss           # FAISS vector index file
│   └── index.pkl             # Serialized embeddings or metadata
│
├── data/                     # Medical PDFs or data files
│
├── .env                      # Environment variables file
└── README.md                 # Project documentation

```
---
## **Contributing** 🤝
Contributions are welcome! If you’d like to improve LetMeHeal, feel free to fork the repo and submit a pull request.

### **Steps to Contribute:**
**Fork the repository**
### **1. Create a new branch:**
```bash
git checkout -b feature-branch
```

### **2. Make your changes and commit:**

```bash
git commit -m "Added new feature"
```
### **3. Push to the branch:**
```bash
git push origin feature-branch
```
### **Open a Pull Request**
---
## **License** 📜
This project is licensed under the MIT License.

💡 Developed with ❤️ by [Mohammad Hashim](https://github.com/mohammadhashim135/Let-Me-Heal)

