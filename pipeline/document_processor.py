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
        chroma_client = chromadb.PersistentClient(path="../knowledgebase")
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

    def generate_embeddings(self, files):
        """
        Generates sentence embeddings for document chunks using a
        SentenceTransformer model.
        """
        embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        for file in files:
            file["embeddings"] = embeddings_model.encode(
                file["texts"], show_progress_bar=True
            )
        return files

    def save_to_chroma(self):
        """
        Coordinates loading, embedding, and storing processed document
        chunks into the ChromaDB collection.
        """
        files = self.generate_embeddings(self.load_files())
        for file in files:
            metadatas = [
                {"source": file["source"], "chunk_id": i}
                for i in range(len(file["texts"]))
            ]
            # ids must be unique across all files
            base_id = file["source"].replace(".pdf", "")
            ids = [f"{base_id}_{i}" for i in range(len(file["texts"]))]

            self.collection.add(
                embeddings=file["embeddings"].tolist(),
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
            model="meta-llama/Llama-3.1-8B-Instruct",
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
        results = self.collection.query(
            query_texts=[self.hyde_query(query)],
            n_results=4,
            include=["documents", "distances"],
        )
        return results
