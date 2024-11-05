# %%
import torch
from multiprocessing import Pool

coordinates = torch.load("/home/bioscience/dev/DeepInteract/features/L5001v1_ligand2/coor_L5001v1_ligand2.pth")


# %% make an adjacency matrix by distance
def make_adjacency_by_distance(coor):
    n = coor.shape[0]
    adj = torch.zeros(n, n)
    for j in range(n):
        for k in range(j + 1, n):
            d = torch.norm(coor[j] - coor[k])
            adj[j, k] = d
            adj[k, j] = d
    return adj


# %% map the function to the list of coordinates
# Define the function for multiprocessing
def process_coordinates(coor):
    return make_adjacency_by_distance(coor)


# Create a pool of processes
pool = Pool()

# Map the function to the list of coordinates using multiprocessing
adjs = pool.map(process_coordinates, coordinates)

# Close the pool of processes
pool.close()
pool.join()
print(adjs[0].shape)

torch.save(adjs, "/home/bioscience/dev/DeepInteract/features/L5001v1_ligand2/adj_L5001v1_ligand2.pth")
print("DONE: adj")

# %%
