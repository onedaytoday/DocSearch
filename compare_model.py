import torch
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, AutoModelForCausalLM
from torch.nn.functional import cosine_similarity
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


def main():
    MODEL_PATH_OR_NAME_A = r"meta-llama/Llama-3.2-1B"
    MODEL_PATH_OR_NAME_B = r"./saved_models"

    model_a = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_A)

    model_b = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_B)

    cloud_test(model_a,model_b)





def calculate_cos_similarity(embeder, docs, queries):
    q_reps = embeder.encode(queries)
    d_reps = embeder.encode(docs)

    q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
    d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
    return torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

def cloud_test(a, b):
    documents = [
        "Clouds are formed when water vapor in the atmosphere cools and condenses into visible water droplets or ice "
        "crystals.",
        "There are different types of clouds, such as cumulus, stratus, and cirrus, which vary in appearance and "
        "altitude.",
        "Cloud computing refers to the delivery of computing services like storage, databases, and software over the "
        "internet.",
        "Clouds can have a significant impact on weather patterns and climate conditions."
    ]

    prompts = [
        "What causes clouds to form?",
        "Explain the different types of clouds.",
        "What is cloud computing?",
        "How do clouds affect the weather?"
    ]

    print(calculate_cos_similarity(embeder=a, docs=documents, queries=prompts))
    print(calculate_cos_similarity(embeder=b, docs=documents, queries=prompts))


if __name__ == '__main__':
    main()
