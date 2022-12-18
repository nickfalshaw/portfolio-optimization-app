import anvil.server
import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

@anvil.server.callable
def run_PCA(price_der,percent_Explained_Variance,ticker_Array):
  norm_Data = StandardScaler().fit_transform(price_der)
  pca_Model = PCA(percent_Explained_Variance)
  pca_Components = pca_Model.fit_transform(norm_Data)

  #Transform data back to original space
  projected_Data = pca_Model.inverse_transform(pca_Components)

  #Calculate the norm distance between orignal and projected data
  squared_difference = np.square(norm_Data-projected_Data)
  norm_distance_Transposed = np.sqrt(squared_difference.sum(axis=0))
  norm_distance = norm_distance_Transposed.reshape(1,len(norm_distance_Transposed))

  #Reformat into dataframe and locate the min and max portfolios 
  norm_distance_Dataframe = pd.DataFrame(data = norm_distance, index = ['Norm'], columns = ticker_Array)
  ascending_Sort = norm_distance_Dataframe.sort_values(by='Norm',axis=1)
  min_Portfolio = ascending_Sort.iloc[:, :10]
  max_Portfolio = ascending_Sort.iloc[:, len(norm_distance_Transposed)-10:]

  portfolio = min_Portfolio + max_Portfolio

  print(portfolio.columns.tolist())
  
  return portfolio.columns.tolist()
  

