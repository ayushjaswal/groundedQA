import gradio as gr
import os
from pipeline.document_processor import PreprocessDocument
from pipeline.answerer import Answerer
from dotenv import load_dotenv
load_dotenv(override=True)

# global state — initialized once
preprocessor = None
answerer = Answerer()

def ingest_documents(files):
    """
    Triggered when user uploads PDFs
    """
    global preprocessor

    if not files:
        return "No files uploaded."

    # save uploaded files to docs/ folder
    os.makedirs("docs", exist_ok=True)
    for file in files:
        file_name = os.path.basename(file.name)
        with open(f"docs/{file_name}", "wb") as f:
            with open(file.name, "rb") as src:
                f.write(src.read())

    # index them
    preprocessor = PreprocessDocument("docs").save_to_chroma()
    return f"✅ {len(files)} document(s) indexed successfully. You can now ask questions."


def answer_question(question):
    """
    Triggered when user submits a question
    """
    global preprocessor

    if preprocessor is None:
        return "⚠️ Please upload and index documents first."

    if not question.strip():
        return "⚠️ Please enter a valid question."

    results = preprocessor.query_kb(question)
    chunks = results["documents"][0]

    prompt = answerer.builder_prompt(question=question, chunks=chunks)
    response = answerer.answer(prompt=prompt)

    return response


# UI
with gr.Blocks(title="GroundedQA") as app:
    gr.Markdown("# 🏆 GroundedQA")
    gr.Markdown("Upload your documents and ask questions grounded in their content.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⬆️ Upload Documents")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDFs"
            )
            upload_btn = gr.Button("Index Documents", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload status will appear here..."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask a Question")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What is the projection head in SimCLR?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Markdown(label="Answer")

    # wire up
    upload_btn.click(
        fn=ingest_documents,
        inputs=[file_input],
        outputs=[upload_status]
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    app.launch()