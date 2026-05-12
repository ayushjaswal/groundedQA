import pymupdf4llm
import re
from pathlib import Path
import os 
from dotenv import load_dotenv
load_dotenv(override=True)

def extract_text(pdf_path):
    """
    Extracts raw text from a PDF file and converts it into markdown format
    using PyMuPDF4LLM.
    """
    file_mkdn = pymupdf4llm.to_markdown(pdf_path)
    return file_mkdn


def clean_markdown(text):
    """
    Cleans markdown text by removing tables, image placeholders, picture
    text blocks, and excessive newlines.
    """
    # remove markdown tables
    text = re.sub(r'\|.*\|.*\n', '', text)
    # remove image placeholders
    text = re.sub(r'==>.*?<==\n?', '', text)
    # remove picture text blocks
    text = re.sub(r'----- Start of picture text -----.*?----- End of picture text -----', 
                  '', text, flags=re.DOTALL)
    # remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_and_clean(file_path):
    """
    Extracts, cleans, and returns the content of a PDF file (standalone
    for ProcessPoolExecutor compatibility).
    """
    text = clean_markdown(extract_text(str(file_path)))
    return {"source": Path(file_path).name, "content": text}

def process_document(file):
    """
    Extracts, cleans, and recursively splits a document into chunks based on
    markdown headers and size constraints.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    cleaned = extract_and_clean(file)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ]
    )
    header_chunks = header_splitter.split_text(cleaned["content"])

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    final_chunks = char_splitter.split_documents(header_chunks)
    texts = [chunk.page_content for chunk in final_chunks]

    return {"source": cleaned["source"], "texts": texts}




# def hyde_query(question):
#     """
#     Generates a hypothetical factual paragraph answering the question
#     via Llama-3.1-8B-Instruct.
#     """
#     hf_client = InferenceClient()
#     response = hf_client.chat_completion(
#         model=os.getenv("HYPO_MODEL"),
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a engineering tutor assistant. Given a question, write a short factual paragraph that directly answers it, as if you were writing it in a research paper. Do not ask questions back."
#             },
#             {
#                 "role": "user",
#                 "content": question
#             }
#         ],
#         max_tokens=150,
#     )
#     return response.choices[0].message.content