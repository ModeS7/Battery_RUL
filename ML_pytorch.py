import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from libs import models, functions
from datetime import datetime


# Load CSV data into a pandas DataFrame
df = pd.read_csv('data/Battery_RUL_cleaned.csv')
#df = pd.read_csv('data/Battery_RUL.csv')
df = df[df.columns[1:]]  # Remove the first column

# Last column is the target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


#model = models.SimpleNN(10).to(device)
#model = models.NN(512, 0.2).to(device)
#model = models.NeuralNetwork(512).to(device)
#model = models.LSTMNN(32, 1, 0).to(device)
#model = models.GRUNN(32, 1, 0).to(device)
model = models.CNN(16, 4).to(device)
#model = models.TNN(16, 2, 1).to(device)
#model = models.ANN(16).to(device)

print(model)

# Get the current time
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Get the model name
model_name = model.__class__.__name__

# Initialize TensorBoard writer
writer = SummaryWriter(f'runs/{model_name}_{current_time}')

loss1 = nn.MSELoss()
loss2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
epochs = 1500

# Initialize early stopping
early_stopping = functions.EarlyStopping(patience=100, min_delta=0.01)

pbar = tqdm(range(epochs))
for t in pbar:
    train_loss = functions.train(train_dataloader, model, loss1, loss2, optimizer, t, device, writer)
    test_loss = functions.test(test_dataloader, model, loss1, loss2, t, device, writer)
    test_loss_MAE = functions.test(test_dataloader, model, nn.L1Loss(), nn.L1Loss(), t, device, writer)
    test_loss_MSE = functions.test(test_dataloader, model, nn.MSELoss(), nn.MSELoss(), t, device, writer)
    scheduler.step()
    pbar.set_description(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Loss MAE: {test_loss_MAE:.4f}, Test Loss MSE: {test_loss_MSE:.4f}")

    # Check for early stopping
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Done!")
torch.save(model.state_dict(), f'models/{model_name}_{current_time}.pt')  # Save only the state_dict

# Close the TensorBoard writer
writer.close()

model.eval()
X, y = next(iter(test_dataloader))
X, y = X.to(device), y.to(device)
with torch.no_grad():
    pred = model(X)
    predicted, actual = pred[0], y[0]
    print(f'Input: "{X[0]}", Predicted: "{predicted}", Actual: "{actual}"')