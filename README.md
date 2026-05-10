# 🧬 My ScienceQA Grounded RAG Project

👋 **Hey there!** This is a personal hobby project I built to play around with RAG (Retrieval-Augmented Generation) and LLM fine-tuning. 

The main goal here is to answer multiple-choice science questions by retrieving relevant context from documents (like textbooks or PDFs) and then feeding that context into a custom model I fine-tuned. The model is trained to select the correct choice and write out a detailed explanation of why it’s correct.

---

## 🚀 Hugging Face Links

I hosted all the models, custom tokenizer, and datasets I used/trained directly on Hugging Face so they're easy to access:

* **Fine-Tuned Model**: [🤗 ayushjaswal/scienceqa-llama32_it1](https://huggingface.co/ayushjaswal/scienceqa-llama32_it1) (A fine-tuned LLaMA 3.2 Instruct model!)
* **My Custom Tokenizer**: [🤗 ayushjaswal/scienceqa-llama32_it1_tokenizer](https://huggingface.co/ayushjaswal/scienceqa-llama32_it1_tokenizer)
* **My Cleaned Dataset**: [🤗 ayushjaswal/scienceQAcleaned](https://huggingface.co/datasets/ayushjaswal/scienceQAcleaned) (I took the original `derek-thomas/ScienceQA` dataset and cleaned it up for this project).
---

## 🔧 How to Set Up and Run This Locally

If you want to clone this and run it on your own machine, here is how to get started:

### 1. Clone the Repo
```bash
git clone https://github.com/ayushjaswal/scienceqa-project.git
cd scienceqa-project
```

### 2. Create a Virtual Environment
```bash
uv init
uv add -r requirement.txt
uv sync
```
---

## ✍️ Some Notes on the Fine-Tuning

* **Base Model**: LLaMA-3.2-Instruct
* **How I Trained It**: I fine-tuned it using **QLoRA** (4-bit quantization with `bitsandbytes` and `peft`) so that I could train and test it easily without needing super heavy compute or enterprise rigs.
* **The Data**: I noticed the original `derek-thomas/ScienceQA` dataset had some noise such as images and other unstructured data, so I cleaned it up to create `ayushjaswal/scienceQAcleaned`. 

The model performed with the accuracy of 93.4% which was better than the expected accuracy of 76% with the base model on the test dataset. It's response is just one label therefore it is good at multiple choice questions.

Hope you find this project interesting! Feel free to reach out if you have any questions or just want to chat about RAG and model training. 😊