import torch
from tqdm.auto import tqdm
from torch import nn
import pandas as pd
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from torch.utils.data import DataLoader
import requests
from pathlib import Path
import matplotlib.pyplot as plt


# Set device inside the script so it's available as a default value
device = "cuda" if torch.cuda.is_available() else "cpu"

##Make predictions and get model_ 0 results
#building multiple models
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device = device):
  loss, acc =0,0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      X, y = X.to(device), y.to(device) # Corrected: Reassign X and y to their device-moved versions
      y_pred = model(X)

      #accumulate the loss and accuracy value per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    loss /= len(data_loader)
    acc /= len(data_loader)

    return {"model_name" : model.__class__.__name__,
            "model_loss" : loss.item(),
            "model_acc" : acc}

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
  train_loss, train_acc = 0, 0
  model.train()
  # add a loop to loop through training loop
  for batch, (X, y) in enumerate(data_loader):
          X , y = X.to(device), y.to(device)

          #1 do the forward pass
          y_pred = model(X)

          #2 calculate the loss
          loss = loss_fn(y_pred, y)
          train_loss += loss
          train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

          #Optimizer zero grad
          optimizer.zero_grad()

          #loss backward
          loss.backward()

          #optimizer step
          optimizer.step()

  train_loss/=len(data_loader)
  train_acc/=len(data_loader)
  print(f"\n Train Loss:{train_loss:.4f} average accuracy: {train_acc} ")

def test_step(model:torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device : torch.device = device):
  model.eval()
  test_loss, test_acc = 0 , 0
  with torch.inference_mode():
    model.eval()
    for X_test, y_test in data_loader:
      #changing the data to device
      X_test, y_test = X_test.to(device), y_test.to(device)

      # Forward pass
      y_test_pred = model(X_test)

      #calculate the loss
      test_loss += loss_fn(y_test_pred, y_test)
      test_acc += accuracy_fn(y_true=y_test, y_pred=y_test_pred.argmax(dim =1))

    #adjust metrics and print out

    test_loss /= len(data_loader)
    test_acc  /= len(data_loader)
    print(f"Test Loss : {test_loss} | Test accuracy: {test_acc}")

from timeit import default_timer as timer
def train_time(start:float,
               end:float,
               device: torch.device = None):
  total_time = end-start
  print(f"Total Time on {device} : {total_time:.3f} seconds")
  return total_time
