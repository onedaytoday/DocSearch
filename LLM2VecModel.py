from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


class LLM2VecModel:
    def __init__(self, model_id, second_model_id=None, token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )
        self.config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True,
            token=token
        )
        pretrained_model = AutoModel.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.base_model = PeftModel.from_pretrained(
            pretrained_model,
            model_id,
        )

        self.base_model = self.base_model.merge_and_unload()

        model_id = second_model_id if second_model_id else model_id

        self.base_plus_SimCSE = PeftModel.from_pretrained(
            self.base_model, model_id
        )

        self.model = LLM2Vec(self.base_plus_SimCSE, self.tokenizer, pooling_mode="mean", max_length=512)
        print("LV2VEC INIT SUCCESSFUL")

    def encode(self, input):
        return self.model.encode(input)

    def train(self):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        trained_model = get_peft_model(self.base_plus_SimCSE, peft_config)
        trained_model.print_trainable_parameters()

