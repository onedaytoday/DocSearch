
import sys

import torch

from LLM2VecModel import LLM2VecModel

HF_PASSCODE = "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"
MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
MODEL_SECONDARY_ID = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"


def main():
    l2v = LLM2VecModel(model_id=MODEL_ID, second_model_id=MODEL_SECONDARY_ID, token=HF_PASSCODE)


def verify_drivers():
    gil_enabled = False
    CUDA = False
    try:
        gil_enabled = sys._is_gil_enabled()
    except:
        pass
    print("GIL Enable ", gil_enabled)
    try:
        CUDA = torch.cuda.is_available()
    except:
        pass
    print("CUDA Enable ", CUDA)




if __name__ == '__main__':
    verify_drivers()
    main()