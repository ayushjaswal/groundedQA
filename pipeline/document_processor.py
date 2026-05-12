import sys
import os
from pathlib import Path
# sys.path.append(str(Path().resolve().parent))
from controllers.utils import (
    extract_text,
    clean_markdown,
    extract_and_clean,
    process_document,
)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)


class PreprocessDocument:
    """
    Handles ingestion, processing, embedding generation, and vector database
    indexing for PDF documents in a knowledge base.
    """

    def __init__(self, kb_path):
        """
        Initializes database clients, CPU worker pools, and sets the path
        for the PDF knowledge base.
        """
        self.kb_path = kb_path
        self.workers = max(os.cpu_count() - 1, 1)
        self.hf_client = InferenceClient()
        chroma_client = chromadb.Client()
        self.collection = chroma_client.get_or_create_collection(name="knowledgebase")

    def load_files(self):
        """
        Loads and processes PDF documents concurrently using a process pool
        and shows progress.
        """
        files = list(Path(self.kb_path).glob("*.pdf"))
        print(f"Loading {len(files)} files with {self.workers} workers")
        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for result in tqdm(pool.map(process_document, files)):
                results.append(result)
        return results

    def generate_embeddings_huggingface(self, files):
        """
        Generates sentence embeddings for document chunks using a
        SentenceTransformer model.
        """
        embeddings_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_OS"))
        for file in files:
            file["embeddings"] = embeddings_model.encode(
                file["texts"], show_progress_bar=True
            ).tolist()
        return files

    def generate_embeddings_openai(self, files):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for file in files:
            response = client.embeddings.create(
                input=file["texts"],
                model=os.getenv("EMBEDDING_MODEL_OAI")
            )
            # extract embedding vector from each item
            file['embeddings'] = [item.embedding for item in response.data]
        return files

    def save_to_chroma(self):
        """
        Coordinates loading, embedding, and storing processed document
        chunks into the ChromaDB collection.
        """
        if os.getenv("ENV_TYPE") == "DEV":
            files = self.generate_embeddings_huggingface(self.load_files())
        else:
            files = self.generate_embeddings_openai(self.load_files())

        for file in files:
            metadatas = [
                {"source": file["source"], "chunk_id": i}
                for i in range(len(file["texts"]))
            ]
            # ids must be unique across all files
            base_id = file["source"].replace(".pdf", "")
            ids = [f"{base_id}_{i}" for i in range(len(file["texts"]))]

            self.collection.add(
                embeddings=file["embeddings"],
                documents=file["texts"],
                ids=ids,
                metadatas=metadatas,
            )
            print(f"Stored {len(file['texts'])} chunks from {file['source']}")
        print(f"Total in collection: {self.collection.count()}")
        return self

    def hyde_query(self, question):
        """
        Uses an LLM to generate a hypothetical answer paragraph to help
        improve vector search relevance (HyDE).
        """
        response = self.hf_client.chat_completion(
            model=os.getenv("HYPO_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a engineering tutor assistant. Given a question, write a short factual paragraph that directly answers it, as if you were writing it in a research paper. Do not ask questions back.",
                },
                {"role": "user", "content": question},
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content

    def query_kb(self, query):
        """
        Retrieves the top-4 most similar document chunks from ChromaDB
        using a HyDE-expanded query.
        """
        # embed the hyde query using the same model as indexing
        hyde = self.hyde_query(query)
        
        if os.getenv("ENV_TYPE") == "DEV":
            embeddings_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_OS"))
            query_embedding = embeddings_model.encode([hyde]).tolist()
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=[hyde],
                model=os.getenv("EMBEDDING_MODEL_OAI")
            )
            query_embedding = [response.data[0].embedding]

        results = self.collection.query(
            query_embeddings=query_embedding,   # <- not query_texts
            n_results=4,
            include=["documents", "distances"],
        )
        return results