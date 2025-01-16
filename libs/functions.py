import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(dataloader, model, loss1, loss2, optimizer, epoch, device, writer):
    model.train()
    for batch in dataloader:
        X, y = batch
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = 0.7 * loss1(pred.squeeze(), y) + 0.3 * loss2(pred.squeeze(), y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log the loss
        writer.add_scalar('Training Loss', loss.item(), epoch)
        return loss

def test(dataloader, model, loss1, loss2, epoch, device, writer):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            test_loss += loss1(pred, y).item() + loss2(pred, y).item()
    test_loss /= num_batches

    # Log the loss
    writer.add_scalar('Test Loss', test_loss, epoch)
    return test_loss
