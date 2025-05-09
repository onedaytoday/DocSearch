import sys

import torch

from LLM2VecModel import LLM2VecModel

HF_PASSCODE = ""
MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
MODEL_SECONDARY_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
Path_to_model = r"C:\Users\Omid\.cache\huggingface\hub\models--meta-llama--Meta-Llama-3.1-8B-Instruct\snapshots" \
                r"\0e9e39f249a16976918f6564b8830bc894c89659"


def main():

    l2v = LLM2VecModel(model_id="meta-llama/Llama-3.2-1B", second_model_id=None, token=HF_PASSCODE, peft=False)
    l2v.fine_tune_unsupervised()


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
