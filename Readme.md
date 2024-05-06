# RagOllama - Adaptive RAG Chatbot
This Adaptive RAG Agent webapp will uses LLM to answer questions about a PDF document. The app uses a graph-based approach to answer questions. The graph is built using LangGraph. 

The RAG technique is used is called Adaptive RAG. Why is this called Adaptive?
1. Routing: Route questions to different retrieval approaches
2. Fallback:  Fallback to web search if docs are not relevant to query
3. Self-correction: Fix answers w/ hallucinations or don’t address question

## Tech Stack
This monorepo has a frontend and backend. The following are the technologies used in this project:-

### Frontend
- [Next.js](https://nextjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Shadcn UI](https://www.shadcn.com/ui)
- [Clerk](https://clerk.dev/)

### Backend
- [Ollama](https://github.com/facebookresearch/llama)
- [Lamma3](https://github.com/facebookresearch/llama/tree/main/llama3)
- [LangChain](https://github.com/hwchase17/langchain)
- [LangChain Community](https://github.com/hwchase17/langchain/tree/main/community)
- [LangGraph](https://github.com/hwchase17/langchain/tree/main/community/langgraph)
- [LangSmith](https://github.com/hwchase17/langchain/tree/main/community/langsmith)
- [Chroma DB](https://www.trychroma.com/)
- [PyPDFLoader](https://github.com/pypdf/PyPDF2)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uppy](https://uppy.io/)

## Getting Started
### Prerequisites
- Python 3.10+
- Node.js 18+
- Ollama (with suitable LLM model installed, in my case I used [llama3:8B](https://github.com/facebookresearch/llama/tree/main/llama3))
- Tavily API Key (for web search)
- LangSmith API Key (optional, for tracing)

### Installation

1. Clone the repository
2. Replace LangSmith API keys in `backend/main.py`
3. Create a `.env` file in `backend` and add the following environment variables:
    - `TAVILY_API_KEY`
4. `cd backend`
5. `pip install -r requirements.txt`
4.`fastapi dev main,py`
1. `cd ../frontend`
2. `npm install`
3. Create a clerk account, and create a clerk project.
4. Add the following keys to `.env` file in `frontend` folder.
    -`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
    -`CLERK_SECRET_KEY`

    -`NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in`
    -`NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up`

    -`NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard`
    -`NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard`
5. `npm run dev`
6. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
7. Create a account by clicking on sign-in button.
8. Upload PDF(s), type your question and click process button.
9. Depending on your specs, it may take a few minutes to process your question.