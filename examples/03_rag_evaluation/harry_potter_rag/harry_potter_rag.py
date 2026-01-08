'''
RAG

Building the Vector Database
PDF -> Text Parsing (PyMuPDF) -> Chunking -> embeddings (OpenAI) -> Vector DB (ChomaDB)

Retrieval
query -> embedding -> vector search -> relevant docs

Augmented

Generation

'''
from parse_pdf_pages import parse_pdf_pages
from chunking import chunk_text
from create_embeddings import create_embeddings
import chromadb
from litellm import completion
from dotenv import load_dotenv

load_dotenv()


client = chromadb.PersistentClient(path="./chroma_db")

def add_chunk_to_chroma_db(chunk, embedding, chunk_index, source):
    chunk_id = f"{source}_chunk_{chunk_index}"
    collection = client.get_or_create_collection(name="documents")

    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": source, "chunk_index": chunk_index}]
    )

    return chunk_id

def add_document_to_vector_db(pdf_path):
    pages_data = parse_pdf_pages(pdf_path)
    full_text = "\n\n".join([page['text'] for page in pages_data if page['text'].strip()])

    chunks = chunk_text(full_text)

    embeddings = create_embeddings(chunks)

    # Extract document name from path for unique IDs
    doc_name = pdf_path.split('/')[-1].replace('.pdf', '')

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        add_chunk_to_chroma_db(chunk, embedding, idx, doc_name)

    print("done...")

    return len(chunks)


def retrieve_from_vector_db(query):
    collection = client.get_or_create_collection(name="documents")
    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    return results


def augment(retrieved_docs, query):
    user_prompt = f"<context>{retrieved_docs}</context>\n\nBased on the above context, answer: {query}"
    return user_prompt


def generate(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": user_prompt}
    ]

    response = completion(
        model="gpt-4o-mini",
        messages=messages,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    pdf_path = "/Users/rakeshbhugra/code/qure/workshop/rag_workshop/input_data/harrypotter.pdf"
    num_chunks = add_document_to_vector_db(pdf_path)
    print(f"Total chunks added to vector DB: {num_chunks}")

    # query = "What is the address where the Dursleys live?"
    query = "how to write fastapi basic code"

    # Retrieval
    results = retrieve_from_vector_db(query)
    retrieved_docs = "\n\n".join(results['documents'][0])
    print(f"Retrieved Documents: {retrieved_docs[:300]}...")

    # Augmentation
    user_prompt = augment(retrieved_docs, query)
    print(f"Augmented Prompt: {user_prompt[:300]}...")

    # Generation
    answer = generate(user_prompt)
    print(f"Generated Answer: {answer}")