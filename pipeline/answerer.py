# PreprocessingPipeline -> Answering
import os
from huggingface_hub import InferenceClient
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

PROMPT = """
You are a scientific assistant. Your role is to understand the query, take the required context retrieved from knowledge base and provide the answer.
If the context is None reply 'There's no information regarding that in the KB' and nothing else.
Understand the prompt, understand the context and also provide easy to understand examples."""


class Answerer:
    def __init__(self):
        self.hf_client = InferenceClient()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def builder_prompt(self, question, chunks=None):
        context = "\n\n".join(chunks) if chunks is not None else "None"
        prompt = f"""
    {PROMPT}

    CONTEXT:
    {context}

Question:
{question}
        """
        return prompt

    def answer(self, prompt):
        if os.getenv("ENV_TYPE") == "DEV":
            response = self.hf_client.chat_completion(
                model=os.getenv("ANSWER_MODEL_HF"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content
        else:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("ANSWER_MODEL_OAI"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content

