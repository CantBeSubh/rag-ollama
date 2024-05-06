from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import pprint


question = ""

### Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


load_dotenv()

local_llm = "llama3"
tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(k=3, tavily_api_key=tavily_api_key)

app = FastAPI()

# Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow requests from any origin (you might want to restrict this in production)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
temp_dir = os.path.join(os.getcwd(), "temp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

llm = ChatOllama(model=local_llm, temperature=0)
llm_json = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions related to a software enginner's Resume like skills, years of experience, and education. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explaination. \n
        Question to route: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm_json | JsonOutputParser()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm_json | JsonOutputParser()

prompt = hub.pull("rlm/rag-prompt")

rag_chain = prompt | llm | StrOutputParser()

prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt | llm_json | JsonOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm_json | JsonOutputParser()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the initial and formulate an improved question. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the initial question: \n\n {question}. Improved question with no preamble:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_rewriter = prompt | llm | StrOutputParser()


class Question(BaseModel):
    ques: str


@app.post("/question/")
async def question(req: Question):
    ques = req.ques
    if ques == "":
        raise HTTPException(status_code=400, detail="No question was provided.")
    try:
        global question
        question = ques
        return {"status": "success", "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    print("QUESTION", question)
    # Raise an error if no files were uploaded
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    # Extract the question headers

    docs = []

    # MARK: LOAD PDF FILES
    for file in files:
        contents = await file.read()
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(f"{temp_file_path}", "wb") as f:
            f.write(contents)

        try:
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
            docs.extend(data)
            print(f"Data loaded for {file.filename}")
        except Exception as e:
            print(f"Failed to load {file.filename}: {str(e)}")
        finally:
            os.remove(temp_file_path)

    text_chunks = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    def retrieve(state):
        """
        Retrieve documents
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
        }

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """
        Transform the query to produce a better question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """
        Web search based on the re-phrased question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    ### Edges ###

    def route_question(state):
        """
        Route question to web search or RAG.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = question_router.invoke({"question": question})
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.
        Args:
            state (dict): The current graph state
        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.
        Args:
            state (dict): The current graph state
        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    ### Workflow ###

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    engine = workflow.compile()
    for output in engine.stream({"question": question}):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
        print("\n---\n")
    # Final generation
    print("FINAL OUTPUT: ", value["generation"])
    return value["generation"]
