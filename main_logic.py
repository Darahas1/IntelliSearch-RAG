import os
import warnings
from dotenv import load_dotenv

# --- MODERN IMPORTS (LangChain v1.0+) ---
# Text Splitters now live in their own package
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Loaders and Vectorstores are in community
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
# Model integrations
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# --- CLASSIC IMPORTS (The fix for your error) ---
# These moved to 'langchain_classic' in v1.0
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

def setup_environment():
    """Validates and loads environment variables."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file.")
        
    return groq_api_key, tavily_api_key

def load_and_split_document(file_path: str):
    """Loads a file and splits it into chunks for the vector store."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()
        
        # Optimized chunk sizes for RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error processing file: {e}")

class HybridRAG:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        # 1. Initialize LLM
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )
        
        # 2. Initialize Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 3. Initialize Web Search Tool
        self.web_search_tool = TavilySearchResults(
            tavily_api_key=tavily_api_key,
            max_results=3
        )
        
        self.vector_store = None
        self.retriever = None

    def setup_document_retriever(self, documents):
        """Builds the vector store and advanced retriever chain."""
        if not documents:
            return
        
        # A. Create Vector Store
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        
        # B. Create Base Retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # C. Multi-Query Retriever (Using langchain_classic)
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )
        
        # D. Contextual Compression (Using langchain_classic)
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever
        )

    def get_web_search_results(self, query: str):
        """Fetches live data from the web using Tavily."""
        try:
            results = self.web_search_tool.invoke({"query": query})
            if not results:
                return "No relevant web search results found."
            
            formatted_results = []
            for res in results:
                url = res.get('url', 'N/A')
                content = res.get('content', 'N/A')
                formatted_results.append(f"Source: {url}\nContent: {content}")
                
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Web search failed: {e}"

    def ask(self, query: str):
        """Main RAG pipeline."""
        
        # 1. Retrieve Document Context
        document_context = ""
        if self.retriever:
            try:
                retrieved_docs = self.retriever.invoke(query)
                if retrieved_docs:
                    document_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                else:
                    document_context = "No relevant information found in the uploaded document."
            except Exception as e:
                # Fallback if compression fails (common with very long docs)
                document_context = f"Error during retrieval: {e}"
        else:
            document_context = "No document uploaded."

        # 2. Retrieve Web Context
        web_context = self.get_web_search_results(query)

        # 3. Final Synthesis Prompt
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are IntelliSearch, an advanced research assistant. 
            Answer the user's question using the provided Document Context and Web Context.
            
            **User Question:** {question}
            
            ---
            **📂 Document Context:**
            {document_context}
            
            ---
            **🌐 Web Context:**
            {web_context}
            ---
            
            **Answer:**
            """
        )
        
        chain = prompt_template | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": query,
            "document_context": document_context,
            "web_context": web_context
        })
        
        return {
            "answer": response,
            "document_context": document_context,
            "web_context": web_context
        }