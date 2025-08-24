import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline


class T5Model(nn.Module):

    def __init__(self, model_path="./pretrained-models/t5-model"):
        super(T5Model, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids, attention_mask, labels):
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)["loss"]
        return loss

    def generate(self, sentence, tokenizer, device, max_length=200):
        generator = Text2TextGenerationPipeline(self.model, tokenizer)
        generator.device = device
        result = generator(sentence, max_length=max_length)[0]["generated_text"].replace(" ", "")
        return result
