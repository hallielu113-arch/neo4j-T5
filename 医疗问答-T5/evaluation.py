import torch
import jieba
import numpy as np
from tqdm import tqdm
from rouge_chinese import Rouge
from nets import T5Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline


def get_test_result(model_name, root="data/test.txt"):
    device = torch.device("cuda")
    rouge = Rouge()
    if model_name.lower() == "t5":
        model = T5Model()
        model.load_state_dict(torch.load("models/t5_epoch5.pth"))
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("pretrained-models/t5-model")
    else:
        raise KeyError("model_name must be T5")

    model = model.to(device=device)
    rouge1, rouge2, rougel = {"precision": [], "recall": [], "f1": []}, {"precision": [], "recall": [], "f1": []}, {
        "precision": [], "recall": [], "f1": []}
    with open(root, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()[:10]):
            line_split = line.strip().split("\t")
            news, summary = line_split[0], line_split[-1]
            summary_pred = model.generate(news, tokenizer=tokenizer, device=device, max_length=500)
            res = rouge.get_scores(" ".join(jieba.cut(summary)), " ".join(jieba.cut(summary_pred)))[0]

            rouge1["precision"].append(res["rouge-1"]["p"])
            rouge1["recall"].append(res["rouge-1"]["r"])
            rouge1["f1"].append(res["rouge-1"]["f"])

            rouge2["precision"].append(res["rouge-2"]["p"])
            rouge2["recall"].append(res["rouge-2"]["r"])
            rouge2["f1"].append(res["rouge-2"]["f"])

            rougel["precision"].append(res["rouge-l"]["p"])
            rougel["recall"].append(res["rouge-l"]["r"])
            rougel["f1"].append(res["rouge-l"]["f"])

    rouge1["precision"] = np.mean(rouge1["precision"])
    rouge1["recall"] = np.mean(rouge1["recall"])
    rouge1["f1"] = np.mean(rouge1["f1"])

    rouge2["precision"] = np.mean(rouge2["precision"])
    rouge2["recall"] = np.mean(rouge2["recall"])
    rouge2["f1"] = np.mean(rouge2["f1"])

    rougel["precision"] = np.mean(rougel["precision"])
    rougel["recall"] = np.mean(rougel["recall"])
    rougel["f1"] = np.mean(rougel["f1"])
    print(model_name)
    print("ROUGE-1")
    print(rouge1)
    print("-" * 50 + ">")

    print("ROUGE-2")
    print(rouge2)
    print("-" * 50 + ">")

    print("ROUGE-l")
    print(rougel)
    print("-" * 50 + ">")


if __name__ == '__main__':
    get_test_result(model_name="t5")
