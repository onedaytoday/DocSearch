import time
import unittest

from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel


class GivenCodeLLM2VecTest(unittest.TestCase):
    def test_given_code(self):
        # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only
        # LLMs. MNTP LoRA weights are merged into the base model.
        tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
        )
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the
        # final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
        model = PeftModel.from_pretrained(
            model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
        )

        # Wrapper for encoding and pooling operations
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        print(l2v.encode(["Hello World"]))


if __name__ == '__main__':
    unittest.main()
