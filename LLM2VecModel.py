from llm2vec import LLM2Vec
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, AutoModelForCausalLM, TextDataset
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from dataset import DataCollatorForMmapedDataset, MmappedArrowDataset


class LLM2VecModel:
    def __init__(self, model_id, second_model_id=None, token=None, peft=True):

        self.llm2vec = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )
        self.config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True,
            token=token
        )
        if not peft:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=token,
                trust_remote_code=True,
                config=self.config,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.model = self.pretrained_model
            return

        self.pretrained_model = AutoModel.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        base_model = PeftModel.from_pretrained(
            self.pretrained_model,
            model_id,
        )
        print(type(base_model))

        self.model = base_model
        if second_model_id:
            base_model = base_model.merge_and_unload()

            # Base Model plus secondary model
            self.model = PeftModel.from_pretrained(
                base_model, second_model_id
            )
            print(type(self.model))

        self.model = self.model.merge_and_unload()

    def make_llm2vec(self):
        self.llm2vec = LLM2Vec(self.model, self.tokenizer, pooling_mode="mean", max_length=512)
        print("LV2VEC INIT SUCCESSFUL")
        return self.llm2vec

    def encode(self, x):
        if self.llm2vec is None:
            self.make_llm2vec()
        return self.llm2vec.encode(x)

    def test(self):
        self.fine_tune_unsupervised()

    def _default_training_argument(self):
        return TrainingArguments(
            output_dir="./results",  # Output directory
            evaluation_strategy="no",  # Evaluation strategy
            logging_dir="./logs",  # Log directory
            num_train_epochs=2,  # Number of training epochs
            per_device_train_batch_size=8,  # Batch size
            save_steps=5000,  # Save checkpoint every 500 steps
            logging_steps=1000,  # Log every 100 steps
        )

    def prepare_dateset(self):
        data = {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "Hugging Face provides powerful NLP models.",
                "Meta Llama 3 is an advanced LLM.",
                "Fine-tuning helps models adapt to specific tasks.",
                "Artificial intelligence is transforming industries.",
            ]
        }

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict(data)

        # Step 2: Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        return tokenized_dataset

    def fine_tune_unsupervised(self, training_args=None, data_collator=None):
        if training_args is None:
            training_args = self._default_training_argument()

        train_dataset = MmappedArrowDataset("./tokenized_files/tokens.arrow", sft=False)

        print(type(self.tokenizer))

        manuels_path = "text_files/combined.txt"
        test_texts_path = "text_files/test_text.txt"
        train_dataset = TextDataset(tokenizer=self.tokenizer, file_path=manuels_path, block_size=128)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Apply LoRA PEFT configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        # Wrap model with LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            data_collator=data_collator,
            args=training_args,
        )

        trainer.train()
        self.model.merge_and_unload().save_pretrained("./saved_models")
        self.tokenizer.save_pretrained("./saved_models")
