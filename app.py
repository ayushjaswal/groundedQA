from pipeline.answerer import Answerer
from pipeline.document_processor import PreprocessDocument
import os

KB_PATH = os.getenv("KB_PATH")

def main():
    
    preprocessor = PreprocessDocument(KB_PATH).save_to_chroma()
    answerer = Answerer()

    query = input("Enter your query: ")
    if not query.strip():
        print("Please enter a valid query")
        return

    results = preprocessor.query_kb(query)
    chunks = results["documents"][0]

    prompt = answerer.builder_prompt(question=query, chunks=chunks)
    response = answerer.answer(prompt=prompt)

    print(response)

    
if __name__ == "__main__":
    main()
