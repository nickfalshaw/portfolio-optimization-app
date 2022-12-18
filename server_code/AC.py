import anvil.server
import numpy as np
from sklearn import preprocessing

@anvil.server.callable
def ac_bump_function(price_der):
      price_der = np.array(price_der)
  
      pairwise_comparison_table = []
      norm_data = preprocessing.normalize((price_der))
      
      for i in range(price_der.shape[0]):
          pairwise_comparison_table.append \
          ([np.exp(-1/(1-(np.linalg.norm(norm_data[i,:]-row)/np.sqrt(2))**2)) \
          if np.linalg.norm(norm_data[i,:]-row)<np.sqrt(2) \
              else 0 for row in norm_data])
      
      for row in range(len(pairwise_comparison_table)):
          for element in range(len(pairwise_comparison_table[row])):
              norm_data[row,:] = norm_data[row,:] + \
                  ((1-pairwise_comparison_table[row][element]) \
                  * norm_data[element,:])
          norm_data[row,:] = norm_data[row,:]/np.sqrt(np.einsum('i,i',norm_data[row,:],norm_data[row,:]))
      return norm_data

## Find Closest Pairwise Vectors 
def computeClosenesses(nparray2d):
  norms = np.zeros(nparray2d.shape[1])
  closest = np.zeros(nparray2d.shape[1])
  ids = np.zeros(nparray2d.shape[1])
  for x in range(0,nparray2d.shape[1]):
    norms=np.zeros(nparray2d.shape[1])
    for y in range(0,nparray2d.shape[1]):
      norms[y]= np.linalg.norm(nparray2d[:,x]-nparray2d[:,y])
    norms[norms==0] = np.inf
    closest[x] = min(norms)
    ids[x]=idx1d(norms,closest[x])
  return closest,ids
  
## Indexing
def idx1d(nparray,lookupvalue):
  for q in range(0,nparray.shape[0]):
    if(nparray[q] == lookupvalue):
      return q 

@anvil.server.callable
def idx1d_callable(nparray,lookupvalue):
  nparray = np.array(nparray)
  for q in range(0,nparray.shape[0]):
    if(nparray[q] == lookupvalue):
      return q 

## Logical Pairwise Matches
def get_binary_link_logical(ids):
  links = np.full(ids.shape[0],False,bool)
  for x in range(0,links.shape[0]):
    spot = int(ids[x])
    if(ids[spot]==x):
      links[x]=True
  return links

## Combine Pairwise Matches
def combine_by_single_linkage(nparray2d,binary_logical,ids,densities):
  newdata = np.empty([nparray2d.shape[0],int(np.round(sum(binary_logical)/2))])
  #newdata = np.empty([nparray2d.shape[0],int(sum(binary_logical)/2)])
  counter=0
  new_densities = np.empty(int(np.round(sum(binary_logical)/2)))
  #new_densities = np.empty(int(sum(binary_logical)/2))
  #untouched_densities = densities[np.where(binary_logical != 1)] #COME BACK TO THIS
  #untouched_data = nparray2d[:,np.where(binary_logical != 1)]
  untouched_densities = densities[np.where(binary_logical == False)] 
  untouched_data = nparray2d[:,np.where(binary_logical == False)]
  checkedvals = np.full(1+int(np.round(sum(binary_logical)/2)),-1)
  relevant = np.where(binary_logical==1)[0]  
  if(len(relevant)==2):
      total_vectors_being_combined = densities[relevant[0]]+densities[relevant[1]]
      coeff1 = densities[relevant[0]]/total_vectors_being_combined
      coeff2 = densities[relevant[1]]/total_vectors_being_combined
      newdata[:,0] = (coeff1*nparray2d[:,relevant[0]]+coeff2*nparray2d[:,relevant[1]]) 
      new_densities[0] = total_vectors_being_combined
  else:
    for x in relevant:
      if((idx1d(checkedvals,int(ids[x])) == None)):
        checkedvals[counter] = x
        #print(checkedvals)
        total_vectors_being_combined = densities[x]+densities[int(ids[x])]
        coeff1 = densities[x]/total_vectors_being_combined
        coeff2 = densities[int(ids[x])]/total_vectors_being_combined
        newdata[:,counter]=(coeff1*nparray2d[:,x]+coeff2*nparray2d[:,int(ids[x])]) 
        new_densities[counter] = total_vectors_being_combined
        counter=counter+1
  combined_newdata = np.concatenate((untouched_data[:,0,:],newdata),axis=1)
  combined_new_densities = np.concatenate((untouched_densities,new_densities),axis = 0)
  return combined_newdata,combined_new_densities

  ## Get Distance Traveled in the dendrogram this iteration
def compute_step_distance(closest,binary_logical,densities):
    if(sum(binary_logical*densities)==0):
        return -1
    else:
        return sum(closest[binary_logical])/(2*sum(binary_logical*densities))
      
@anvil.server.callable
def AgCl(data, StepDists, IterCount, densities):
  data = np.array(data) #<--------------------- why is this necessary?
  StepDists = np.array(StepDists)  #<--------------------- why is this necessary?
  densities = np.array(densities)  #<--------------------- why is this necessary?
  closest,ids = computeClosenesses(data)
  links = get_binary_link_logical(ids)
  combined_newdata,combined_new_densities = combine_by_single_linkage(data,links,ids,densities)
  StepDists = np.append(StepDists, np.array([[0,0]]), axis=0)
  StepDists[IterCount,0] = compute_step_distance(closest,links,densities)
  StepDists[IterCount,1] = combined_newdata.shape[1]
  if(StepDists[IterCount,1]>1):
    return AgCl(combined_newdata,StepDists,IterCount+1,combined_new_densities)
  else:
    return StepDists #,newdata


@anvil.server.callable
def select_clusters(cluster_labels1d,cutoff_percentage):
  #Getting the percentages of the index that each cluster makes up
  cluster_labels1d = np.array(cluster_labels1d)
  unique_numbers = list(set(cluster_labels1d))
  percentages = []
  for i in unique_numbers:
    percentages.append(np.mean(cluster_labels1d==i)) 
  #print(percentages)
  #Choosing the top x% of stock clusters
  reliable_clusters = np.array(percentages)>=cutoff_percentage
  chosen_clusters = [x for x, y in zip(unique_numbers, reliable_clusters) if y]
  return chosen_clusters

@anvil.server.callable
def get_centers(data, chosen_clusters, cluster_labels):
  data = np.array(data)
  cluster_labels = np.array(cluster_labels)
  representatives = np.empty([data.shape[0],len(chosen_clusters)])
  index = 0
  for x in chosen_clusters:
      number_of_vectors=cluster_labels[cluster_labels==x].shape[0]
      proxy= np.sum(data[:,cluster_labels==x],axis=1)/number_of_vectors
      representatives[:,index]=proxy
      index=index+1
  return representatives

@anvil.server.callable
# Get closest stocks to centers 
def closest_stocks_to_centers(data, cluster_centers, cluster_labels, chosen_clusters, num_winners):
  data = np.array(data)
  cluster_centers = np.array(cluster_centers)
  cluster_labels = np.array(cluster_labels)
  index = 0
  restricted_chosen_clusters=[]
  for y in chosen_clusters:
      if(sum(cluster_labels==y)>=num_winners):
          restricted_chosen_clusters.append(y)
  
  picked = np.full((len(restricted_chosen_clusters),num_winners),-1)
  
  for x in restricted_chosen_clusters:
      rankings=[]
      proxy=data[:,cluster_labels==x]
      clusters_in_data_indexes = np.asarray(np.where(cluster_labels==x)).flatten()
      for i in range(proxy.shape[1]):
          rankings.append(np.linalg.norm(proxy[:,i]-cluster_centers[:,index]))
      for k in range(picked.shape[1]):
          spot = rankings.index(min(rankings))
          spot_in_data=clusters_in_data_indexes[spot]
          picked[index,k]=spot_in_data
          rankings[spot]=np.inf 
      index=index+1       
  return picked