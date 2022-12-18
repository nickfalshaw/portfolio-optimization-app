import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Define encoder class 
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        #Initialize
        super(Encoder, self).__init__()
        
        #Data input layer feeds to 100 nodes
        #Try more nodes add more linear layers
        #Try some convoluditon layers (1d time series)
        #Max pull layer 
        #Relu linear layer 
        self.linear1 = nn.Linear(251, 100)
        #self.linear1 = nn.Conv1d(251,100,1)
        #Define split second layer one for mean and one for standard deviation
        self.linear21 = nn.Linear(100, latent_dims)
        self.linear22 = nn.Linear(100, latent_dims)

        #Define loss starting at zero
        self.kl = 0

        #Define initial sample distribution (basic normal distribution)
        self.N = torch.distributions.Normal(0, 1)

        #Force the distrubution to sample on the gpu could be an issue 
        #self.N.loc = self.N.loc.device() 
        #self.N.scale = self.N.scale.device()


    def forward(self, x):
        #Initialize layer 1
        #Try different learning rates 
        x = F.relu(self.linear1(x))

        mu =  self.linear21(x)
        sigma = torch.exp(self.linear22(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = -.5*torch.sum(-sigma**2 - mu**2 + torch.log(sigma) + 1)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 100)
        self.linear2 = nn.Linear(100, 251)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z)) 
        return z

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, x_hat_stored, loss_stored, avg_loss_stored, epochs):
    #try different learning rates. Smaller learning rates 
    #Weight decay addition to optimizer 
    opt = torch.optim.Adam(autoencoder.parameters(),1e-4)
    for epoch in range(epochs):
        i = 0
        for x in data:
            #Define device to process on
            x = x.to(device) 

            #Resets Gradients 
            opt.zero_grad()

            x_hat = autoencoder(x.float())
            x_hat_stored[i,0:251] = x_hat
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss_stored[i] = loss
            loss.backward()
            opt.step()
            i = i+1
        avg_loss_stored[epoch] = loss_stored.mean()
    return avg_loss_stored,x_hat_stored



@anvil.server.callable
def run_VAE(price_der,epochs,device,stock_Tokens):
  while (len(price_der)>251):
    price_der = np.delete(price_der,len(price_der)-1,axis = 0)

  norm_Data = StandardScaler().fit_transform(price_der)

  latent_dims = 2
  
  data_Transpose = np.transpose(norm_Data)
  processingDataloader = torch.utils.data.DataLoader(data_Transpose, 1, shuffle = False)

  x_hat_stored = torch.from_numpy(np.zeros((len(data_Transpose),251)))
  loss_stored = torch.from_numpy(np.zeros((len(data_Transpose),1)))
  avg_loss_stored = torch.from_numpy(np.zeros((epochs,1)))

  autoencoder = Autoencoder(latent_dims).to(device) # GPU

  avg_loss_stored,x_hat_stored = train(autoencoder, processingDataloader, x_hat_stored, loss_stored, avg_loss_stored, epochs)

  #Extract reconstructed data from autoencoder
  data_Reconstructed = np.transpose(x_hat_stored.detach().numpy())

  #Calculate the norm distance between orignal and projected data
  squared_difference = np.square(norm_Data-data_Reconstructed)
  norm_distance = np.sqrt(squared_difference.sum(axis=0))
  norm_distance = norm_distance.reshape(1,len(data_Transpose))

  #Reformat into dataframe and locate the min and max portfolios 
  norm_distance_Dataframe = pd.DataFrame(data = norm_distance, index = ['Norm'], columns = stock_Tokens)
  ascending_Sort = norm_distance_Dataframe.sort_values(by='Norm',axis=1)
  min_Portfolio = ascending_Sort.iloc[:, :10]
  max_Portfolio = ascending_Sort.iloc[:, len(data_Transpose)-10:]

  portfolio = min_Portfolio + max_Portfolio

  return portfolio.columns.tolist()