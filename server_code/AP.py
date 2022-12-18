import anvil.server
import numpy as np
from sklearn import preprocessing

@anvil.server.callable
def ap_bump_function(price_der):
  price_der = np.array(price_der)
  pairwise_comparison_table = []
  norm_data = preprocessing.normalize(price_der)
  
  for i in range(price_der.shape[0]):
      pairwise_comparison_table.append(\
          [\
          np.exp(-1/(1-(np.linalg.norm(norm_data[i,:]-row)/np.sqrt(2))**2))\
          if np.linalg.norm(norm_data[i,:]-row)<np.sqrt(2) \
          else 0 \
          for row in norm_data \
          ])
  
  for k in range(price_der.shape[0]):
      for j in range(price_der.shape[0]):
          if i!=j:
              norm_data[i,:]=(1-pairwise_comparison_table[i][j])*norm_data[i,:]
  return norm_data