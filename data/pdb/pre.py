# %%
import torch
import pickle

#%%
path = "/home/bioscience/dev/DeepInteract_Recomb/features/all/"
# 예시 데이터: 실제로는 파일에서 불러오거나 데이터를 생성해야 함
nodes = torch.load(path + "node.pth")  # 노드 정보 (one_hot_nodes)
#%%
adjs = torch.load(path + "adj.pth")  # 어드제이선시 매트릭스 (adjs)
#%%
targets = torch.load(path + "target_labels.pth")  # 타겟 레이블 (targets)
#%%
ligand = torch.load(path + "mol.pth")  # Ligand 정보 (ligand)
#%%
print(len(nodes))
print(len(adjs))
print(len(targets))
print(len(ligand))
#%%
print(len(nodes[0]))
print(len(adjs[0]))
print(len(targets[0]))
print(len(ligand[0]))

#%%
# 데이터를 저장할 딕셔너리
data_dict = {
    "one_hot_nodes": nodes,  # 노드 정보 저장
    "adjs": adjs,  # 어드제이선시 매트릭스 저장
    "targets": targets,  # 타겟 레이블 저장
    "ligand": ligand,  # Ligand 정보 저장
}
# %%
# pickle을 사용하여 데이터를 저장
with open("/home/bioscience/dev/DeepInteract_Recomb/features/all/pdbbind_data.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("데이터 저장 완료: dataset/pdbbind_data.pkl")
# %%
import torch
import pickle
import os

# 데이터 로드 경로 및 저장 경로
path = "/home/bioscience/dev/DeepInteract_Recomb/features/all/target_labels.pth"
save_dir = "/home/bioscience/dev/DeepInteract_Recomb/features/from_pre/"

# 타겟 레이블 데이터 로드
target_labels = torch.load(path)

# 데이터 크기 확인
total_size = len(target_labels)
split_size = 4000  # 4000개씩 나눔

# 저장 경로가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 데이터를 4000개씩 나누어 저장
part_number = 0
for i in range(0, total_size, split_size):
    # 각 파트의 시작과 끝 인덱스를 설정
    end_idx = min(i + split_size, total_size)  # total_size를 초과하지 않도록 처리

    # 부분적으로 데이터를 저장
    with open(f"{save_dir}target_labels_{i}-{end_idx}.pkl", "wb") as f:
        pickle.dump(target_labels[i:end_idx], f)

    print(f"Saved part {part_number}: {i} to {end_idx} ({end_idx - i} items)")
    part_number += 1

print(f"총 {part_number}개의 파일로 분할 저장 완료")

# %%
import torch
import pickle
import os

# 데이터 로드 경로 및 저장 경로
path = "/home/bioscience/dev/DeepInteract_Recomb/features/all/"
save_dir = "/home/bioscience/dev/DeepInteract_Recomb/features/pdb_split/"

# 분할 저장할 파일 이름
file_names = ["node.pth", "adj.pth", "target_labels.pth", "mol.pth"]
split_size = 2000  # 4000개씩 나눔

# 저장 경로가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 각 파일에 대해 데이터를 나누어 저장하는 함수
def split_and_save(file_name, feature_name):
    # 파일 로드
    data = torch.load(os.path.join(path, file_name))
    total_size = len(data)
    
    part_number = 0
    for i in range(0, total_size, split_size):
        # 각 파트의 시작과 끝 인덱스를 설정
        end_idx = min(i + split_size, total_size)  # total_size를 초과하지 않도록 처리
        
        # 부분적으로 데이터를 저장
        split_file_name = f"{save_dir}{feature_name}_{i}-{end_idx}.pkl"
        with open(split_file_name, "wb") as f:
            pickle.dump(data[i:end_idx], f)
        
        print(f"Saved {feature_name} part {part_number}: {i} to {end_idx} ({end_idx - i} items)")
        part_number += 1

    print(f"총 {part_number}개의 {feature_name} 파일로 분할 저장 완료")

# 각 파일에 대해 분할 저장 수행
split_and_save("node.pth", "nodes")
split_and_save("adj.pth", "adjs")
split_and_save("target_labels.pth", "targets")
split_and_save("mol.pth", "ligand")

# %%
