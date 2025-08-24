import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def get_datasets():
    with open("医学问答数据集/train_datasets.jsonl", "r", encoding="utf-8") as f:
        json_objs = f.readlines()
    with open("data/train.txt", "w", encoding="utf-8") as f:
        for json_obj in tqdm(json_objs):
            json_obj = json.loads(json_obj)
            content = json_obj["questions"][0][0]
            summary = json_obj["answers"][0]
            f.write(content + "\t" + summary + "\n")

    with open("医学问答数据集/validation_datasets.jsonl", "r", encoding="utf-8") as f:
        json_objs = f.readlines()
    with open("data/test.txt", "w", encoding="utf-8") as f:
        for json_obj in json_objs:
            json_obj = json.loads(json_obj)
            content = json_obj["questions"][0][0]
            summary = json_obj["answers"][0]
            f.write(content + "\t" + summary + "\n")


class T5Generator(Dataset):

    def __init__(self, root, tokenizer, max_length):
        super(T5Generator, self).__init__()
        self.root = root
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.news, self.summaries = self.get_datasets()

    def __getitem__(self, item):
        news, summary = self.news[item], self.summaries[item]
        token_info1 = self.tokenizer(news, max_length=self.max_length, truncation=True, padding="max_length")
        token_info2 = self.tokenizer(summary, max_length=self.max_length, truncation=True, padding="max_length")
        input_ids = token_info1["input_ids"]
        attention_mask = token_info1["attention_mask"]
        labels = token_info2["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(labels)

    def __len__(self):
        return len(self.summaries)

    def get_datasets(self):
        news, summaries = [], []
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line_split = line.strip().split("\t")
                news.append(line_split[0])
                summaries.append(line_split[-1])

        return news, summaries


if __name__ == '__main__':
    get_datasets()
