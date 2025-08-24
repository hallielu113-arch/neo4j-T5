import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import T5Model
from datasets import T5Generator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def train_t5_model(model_name):
    print(model_name)
    device = torch.device("cuda")
    model_path = "pretrained-models/t5-model"
    model = T5Model().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_loader = DataLoader(T5Generator(root="data/train.txt", tokenizer=tokenizer, max_length=300), batch_size=8,
                              shuffle=True)
    val_loader = DataLoader(T5Generator(root="data/test.txt", tokenizer=tokenizer, max_length=300), batch_size=32,
                            shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=5)
    train_losses, val_losses = [], []
    min_loss = 10000
    for epoch in range(10):
        train_loss = train_t5_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        vaL_loss = get_t5_val_result(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(vaL_loss)
        print(f"epoch:{epoch + 1},train_loss:{train_loss},val_loss:{vaL_loss}")
        if vaL_loss < min_loss:
            min_loss = vaL_loss
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"models/{model_name}_epoch{epoch + 1}.pth")

    plot_loss(train_losses, model_name=model_name)


def train_t5_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    data = tqdm(train_loader)
    losses = []
    for batch, (x, y, z) in enumerate(data):
        input_ids, attention_mask, labels = x.to(device), y.to(device), z.to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data.set_description_str(f"epoch:{epoch + 1},batch:{batch + 1},loss:{loss.item()}")
    scheduler.step()

    return float(np.mean(losses))


def get_t5_val_result(model, val_loader, device):
    model.eval()
    data = tqdm(val_loader)
    losses = []
    with torch.no_grad():
        for batch, (x, y, z) in enumerate(data):
            input_ids, attention_mask, labels = x.to(device), y.to(device), z.to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(loss.item())

    return float(np.mean(losses))


def plot_loss(train_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses[:80]) + 1), train_losses[:80], "r")
    plt.title(f"images/{model_name}_loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"images/{model_name}_loss-epoch.jpg")
    # plt.show()


if __name__ == '__main__':
    train_t5_model(model_name="t5")
