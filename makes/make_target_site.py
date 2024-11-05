# %% Imports
import os
import torch
from IPython.display import display
from utils.pipe import Pipe
from utils.prep import (
    make_sequence_and_coordinates,
)
import pickle
import logging

nproc = os.cpu_count() - 2

# Define the range of data entries to process
starting_number = 0
ending_number = 19300

# Define file paths and formats
MERGED_DIR = "/home/bioscience/datasets/deepinteract-dataset/merged/"
#PROTEIN_FILE_FORMAT = MERGED_DIR + "proteins/{}_protein.pdb"
POCKET_FILE_FORMAT = MERGED_DIR + "pockets/{}_pocket.pdb"

path_name = "/home/bioscience/dev/DeepInteract_Recomb/features/all/name.pth"

with open(path_name, 'rb') as file:
    names = torch.load(file)

names = names[starting_number:ending_number]

#proteins = [PROTEIN_FILE_FORMAT.format(name) for name in names]
pockets = [POCKET_FILE_FORMAT.format(name) for name in names]
#logging.info(f"{len(names)} data entries to process")

#p_coor = Pipe(pockets, make_coordinates)
#coord_pocket = p_coor.run_multiproc(cores=nproc)

p_coor = Pipe(pockets, make_sequence_and_coordinates)
seq_and_coord = p_coor.run_multiproc(cores=nproc)
sequences, coord_pocket = zip(*seq_and_coord)
logging.info("DONE: sequence and coordinates")

#%%
path_coord = "/home/bioscience/dev/DeepInteract_Recomb/features/all/coor.pth"

with open(path_coord, 'rb') as file:
    coord_protein = torch.load(file)

target = []
for i in range(len(coord_pocket)):
    length = len(coord_protein[i])
    zeros_tensor = torch.zeros(length, dtype=torch.int)
    target.append(zeros_tensor)

#%%
total = []
count_same = 0

for i, (protein, pocket) in enumerate(zip(coord_protein, coord_pocket)):
    for k in range(len(pocket)):  # pocket의 좌표들을 순회
        for index, value in enumerate(protein):  # protein의 좌표들을 순회
            if torch.equal(pocket[k], value):  # pocket 좌표와 protein 좌표 비교
                target[i][index] = 1  # 일치하는 좌표의 인덱스에 대해 target 수정
                break
#%%
total = []
count_same = 0

for i, (protein, pocket) in enumerate(zip(coord_protein, coord_pocket)):
    for k in range(len(pocket)):  # pocket의 좌표들을 순회
        count_same = 0  # 각 k마다 초기화
        #found = False
        for index, value in enumerate(protein):  # protein의 좌표들을 순회
            if torch.equal(pocket[k], value):  # pocket 좌표와 protein 좌표 비교
                target[i][index] = 1  # 일치하는 좌표의 인덱스에 대해 target 수정
                count_same += 1  # 조건이 만족될 때마다 count_same 증가
                found = True
                break  # 일치하는 좌표가 발견되면 내부 루프 종료
        # 일치하는 좌표를 찾지 못하면 인덱스와 그 좌표를 반환
        # if not found:
        #     print(f"{i}: {pocket[k]}")
        total.append(count_same)  # 각 k에 대한 비교 결과를 total에 저장

# %%
# 검증 단계: target[i]의 값이 1인 갯수(합)이 pocket_coords[i]의 길이와 같은지 확인
count = 0
num = []
for i in range(len(target)):
    target_sum = sum(target[i])  # target[i]에서 값이 1인 갯수 합산
    pocket_len = len(coord_pocket[i])  # 해당 포켓의 좌표 개수

    if target_sum == pocket_len:
        #print(f"Data {i}: 검증 성공 (1로 바뀐 좌표 수 = 포켓 좌표 수)")
        count += 0
    else:
        num.append(i)
        print(pocket_len - target_sum)
        count += 1
        #print(f"Data {i}: 검증 실패 (1로 바뀐 좌표 수: {target_sum}, 포켓 좌표 수: {pocket_len})")

print(count)
print(num)

# %%
for i in num:
    if torch.equal(coord_pocket[i][-1], coord_pocket[i][-2]):
        continue
    print(i, coord_pocket[i][-2:])

# %%
# 검증2
for i in range(len(target)):
    if target[i].shape[0] != coord_protein[i].shape[0]:
        print(i)


# %%
for n in num:
    print(names[n])


# %%
print(len(coord_protein[2918]))
print(len(coord_pocket[2918]))
print(coord_protein[2918])
print(coord_pocket[2918])
# %%
target_test = []

length_test = len(coord_protein[2918])
zeros_tensor_test = torch.zeros(length_test, dtype=torch.int)
target_test.append(zeros_tensor_test)

print(target_test)
#%%
count_same = 0
real_index = []

for k in range(len(coord_pocket[2918])):  # pocket의 좌표들을 순회
    for index, value in enumerate(coord_protein[2918]):  # protein의 좌표들을 순회
        # print(coord_protein[2918][index])
        # print(coord_pocket[2918][k])
        if torch.equal(coord_pocket[2918][k], value):  # pocket 좌표와 protein 좌표 비교
            target_test[0][index] = 1
            count_same += 1
            real_index.append(index)
            break

print(count_same)
print(sum(target_test[0]))
print(len(target_test[0]))

# %%
target_real_index = []

for target_i, value in enumerate(target_test[0]):
    if value == 1:
        target_real_index.append(target_i)

#%%
output_file = "target_label.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(target, f)

print(f"Target label saved to {output_file}")
# %%
