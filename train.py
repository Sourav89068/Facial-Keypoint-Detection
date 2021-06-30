import torch
import numpy as np
from model import CNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(outputs=30)
model = model.to(device)
def train(train_loader, valid_loader, model, criterion, optimizer,n_epochs=50, saved_model='model.pt'):
    valid_loss_min = np.Inf
    train_losses = []
    valid_losses = []
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train() 
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['image'].to(device))
            loss = criterion(output, batch['keypoints'].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch['image'].size(0)
        model.eval() 
        for batch in valid_loader:
            output = model(batch['image'].to(device))
            loss = criterion(output, batch['keypoints'].to(device))
            valid_loss += loss.item()*batch['image'].size(0)
        train_loss = np.sqrt(train_loss/len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss/len(valid_loader.sampler.indices))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss            
    return train_losses, valid_losses