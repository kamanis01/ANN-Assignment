import numpy as np
import torch
from classifier import AudioANNClassifier
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader
from dataset_utils import get_datasets

chkpt = torch.load('models/urban8k_ann_librosa_v1.pt', map_location='cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AudioANNClassifier(in_features=16, out_features=10)
model.load_state_dict(chkpt['model_state_dict'])
model = model.to(device)
model.eval()

_, _, test_dataset = get_datasets("statistics.csv")
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, pin_memory=False, num_workers=4)

y_pred = []
y_test = []
with torch.no_grad():
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch = x_test_batch.to(device)
        y_pred_batch = model(x_test_batch).cpu()
        y_pred_batch = torch.softmax(y_pred_batch, dim=1)
        y_pred_batch = torch.argmax(y_pred_batch, dim=1)

        y_pred += y_pred_batch.tolist()
        y_test += y_test_batch.tolist()

y_test = np.array(y_test)
y_pred = np.array(y_pred)

conf_mat = confusion_matrix(y_test, y_pred)
clf_rep = classification_report(y_test, y_pred)

print(conf_mat)
print(clf_rep)
