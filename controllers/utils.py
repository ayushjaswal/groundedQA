import pymupdf4llm
import re
from pathlib import Path

def extract_text(pdf_path):
    """
    A function to extract pdf into markdown. 
    """
    file_mkdn = pymupdf4llm.to_markdown(pdf_path)
    return file_mkdn


def clean_markdown(text):
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
    Standalone function (not a method) — required for ProcessPoolExecutor
    since it can't pickle instance methods
    """
    text = clean_markdown(extract_text(str(file_path)))
    return {"source": Path(file_path).name, "content": text}

def process_document(file):
    """
    Standalone function — ProcessPoolExecutor can't pickle instance methods
    so this must live outside the class
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




def hyde_query(question):
    hf_client = InferenceClient()
    response = hf_client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a engineering tutor assistant. Given a question, write a short factual paragraph that directly answers it, as if you were writing it in a research paper. Do not ask questions back."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content