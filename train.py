import time
import torch
from classifier import AudioANNClassifier
from dataset_utils import get_datasets
from torch.utils.data import DataLoader
from torch import nn, optim

nb_epochs = 5000
batch_size = 4096
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, val_dataset, _ = get_datasets(df_path='statistics.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)

model = AudioANNClassifier(in_features=16, out_features=10).to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_dataset.class_weights, dtype=torch.float32)).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

max_val_acc = -1

for i in range(nb_epochs):
    train_loss = 0
    val_loss = 0
    start = time.time()

    model.train()
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))

    model.eval()
    nb_corrects = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            preds = torch.softmax(outputs, dim=1).max(dim=1)[1]
            nb_corrects += torch.sum(preds == labels).item()

            loss = criterion(outputs, labels)

            val_loss += (loss.item() * data.size(0))

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    val_acc = nb_corrects / len(val_dataset)

    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save({
            'epoch': i + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, "models/urban8k_ann_librosa_v1.pt")

    if i % 100 == 0:
        print(f"Epoch {i} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val. loss: {val_loss:.4f} | "
              f"Val. Acc: {val_acc:.4f} | "
              f"Time: {(time.time() - start):.4f}s")
