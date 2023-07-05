import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def load_docs():
    # Load your documents here
    documents = [
        {"page_content": "stello_EnergyStar Smart Thermostats V2_Comments_0.pdf"}
        # Add more documents as needed
    ]
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_texts = [doc["page_content"] for doc in documents]
    docs = text_splitter.split_documents(doc_texts)
    return docs

def get_similar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Load documents
documents = load_docs()
print(f"Number of documents loaded: {len(documents)}")

# Split documents
docs = split_docs(documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Embed a query
query_result = embeddings.embed_query("Hello world")
print(f"Length of query embedding: {len(query_result)}")

# Initialize Pinecone index
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
index_name = "langchain-demo"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Initialize OpenAI model
model_name = "gpt-3.5"
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# Perform a question answering query
query = "What is the name of Table 01?"
answer = get_answer(query)
print(f"Answer: {answer}")

# Perform similarity search
similar_docs = get_similar_docs(query)
print(f"Similar documents: {similar_docs}")
