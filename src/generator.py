# src/generator.py
# =============================================================================
# Prompting strategies and LLM generation.
# Four strategies compared: zero-shot, few-shot, chain-of-thought, structured.
# Structured prompting selected as it reduces hallucination in small models.
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Prompting strategies ─────────────────────────────────────────────────────

STRATEGIES = {

    "zero_shot": {
        "system": (
            "You are a knowledgeable culinary assistant specialising in East Asian cuisine. "
            "Answer questions accurately and concisely using only the provided context. "
            "If the context does not contain enough information, say so honestly."
        ),
        "template": (
            "Context information:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    },

    "few_shot": {
        "system": (
            "You are a knowledgeable culinary assistant specialising in East Asian cuisine. "
            "Answer questions accurately and concisely using only the provided context. "
            "Follow the style of the examples provided."
        ),
        "template": (
            "Example 1:\n"
            "Context: Miso is a traditional Japanese seasoning produced by fermenting soybeans with salt and koji.\n"
            "Question: What is miso made from?\n"
            "Answer: Miso is made from soybeans fermented with salt and koji. "
            "Fermentation can take weeks to years depending on the type.\n\n"
            "Example 2:\n"
            "Context: Kimchi is a traditional Korean side dish of salted and fermented vegetables.\n"
            "Question: What is kimchi?\n"
            "Answer: Kimchi is a traditional Korean fermented vegetable dish, "
            "most commonly made with napa cabbage seasoned with chilli and garlic.\n\n"
            "Now answer using the context below:\n\n"
            "Context information:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    },

    "chain_of_thought": {
        "system": (
            "You are a knowledgeable culinary assistant specialising in East Asian cuisine. "
            "First identify key facts from the context, then give a clear final answer."
        ),
        "template": (
            "Context information:\n{context}\n\n"
            "Question: {query}\n\n"
            "Key facts from context:\n-\n\n"
            "Final answer:"
        ),
    },

    "structured": {
        "system": (
            "You are a precise culinary assistant specialising in East Asian cuisine.\n"
            "Rules:\n"
            "1. Answer ONLY using information from the provided context\n"
            "2. Keep your answer between 2 and 5 sentences\n"
            "3. Do NOT add information not present in the context\n"
            "4. Do NOT say 'based on the context' or 'the context states'\n"
            "5. Write in clear, direct English"
        ),
        "template": (
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Direct answer (2-5 sentences, using only the context above):"
        ),
    },
}


def build_context(retrieved_chunks, max_words=600):
    """
    Formats retrieved chunks into a numbered context block for the prompt.
    Truncates to max_words to fit within the LLM context window.

    Args:
        retrieved_chunks : list of result dicts from Retriever.retrieve()
        max_words        : maximum total words in context
    Returns:
        formatted context string
    """
    context_parts = []
    word_count    = 0

    for i, result in enumerate(retrieved_chunks, 1):
        chunk_text = result["chunk"]["text"]
        title      = result["chunk"]["doc_title"]
        words      = chunk_text.split()

        remaining = max_words - word_count
        if remaining <= 30:
            break
        if len(words) > remaining:
            chunk_text = " ".join(words[:remaining]) + "..."

        context_parts.append(f"[{i}] Source: {title}\n{chunk_text}")
        word_count += len(chunk_text.split())

    return "\n\n".join(context_parts)


class Generator:
    """
    Wraps Qwen2.5-0.5B-Instruct for RAG answer generation.
    Supports all four prompting strategies.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Args:
            model_name : HuggingFace model name
        """
        self.model_name = model_name
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print(f"Generator loaded on {self.device}")

    def generate(self, query, context, strategy="structured",
                 max_new_tokens=200, temperature=0.3):
        """
        Generates an answer for a query given retrieved context.

        Args:
            query          : question string
            context        : formatted context string from build_context()
            strategy       : one of "zero_shot", "few_shot",
                             "chain_of_thought", "structured"
            max_new_tokens : maximum tokens to generate
            temperature    : sampling temperature (lower = more focused)
        Returns:
            generated answer string
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Choose from {list(STRATEGIES.keys())}")

        strat       = STRATEGIES[strategy]
        system_msg  = strat["system"]
        user_msg    = strat["template"].format(context=context, query=query)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
