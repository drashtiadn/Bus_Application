from fastapi import FastAPI, requests, UploadFile, File
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
import os

app = FastAPI()

folder_path = "db"


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

cached_llm = ChatGroq(model="mixtral-8x7b-32768",temperature=0)


embedding = SentenceTransformer('moka-ai/m3e-base')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
    """
)


@app.post("/ai/")
async def aiPost(request: requests):
    json_content = await request.json()
    query = json_content.get("query")

    response = cached_llm.invoke(query)
    response_answer = {"answer": response}
    return response_answer


@app.post("/ask_pdf/")
async def askPDFPost(request: requests):
    json_content = await request.json()
    query = json_content.get("query")

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1},
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.post("/pdf/")
async def pdfPost(file: UploadFile = File(...)):
    file_name = file.filename
    save_file = os.path.join("pdf", file_name)

    with open(save_file, "wb") as f:
        f.write(await file.read())

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


@app.post("/agent_query/")
async def agentQueryPost(request: requests):
    json_content = await request.json()
    query = json_content.get("query")

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever_tool = create_retriever_tool(vector_store, "langsmith_search", "Search for information about LangSmith. For any questions about LangSmith. You should use this")

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    duckduckgo = DuckDuckGoSearchRun(api_wrapper=duckduckgo_wrapper)

    tools = [retriever_tool, wiki, duckduckgo]

    agent = create_openai_tools_agent(cached_llm, tools, raw_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": query})

    response_answer = {"answer": response}
    return response_answer


def start_app():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    start_app()
