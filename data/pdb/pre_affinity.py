#%%
import torch
import pickle

path = "/home/bioscience/dev/DeepInteract_Recomb/features/all/"
save_path = "/home/bioscience/dev/DeepInteract_Recomb/features/splits/"

# 예시 데이터 불러오기
nodes = torch.load(path + "node.pth", map_location='cpu')
adjs = torch.load(path + "adj.pth", map_location='cpu')
targets = torch.load(path + "target.pth", map_location='cpu')
ligand = torch.load(path + "mol.pth", map_location='cpu')
#%%
# 데이터 크기 확인
total_size = len(nodes)
split_size = 3500
#%%
# 데이터를 4000개씩 나누어 저장
for i in range(0, total_size, split_size):
    data_chunk = {
        "one_hot_nodes": nodes[i:i + split_size],
        "adjs": adjs[i:i + split_size],
        "targets": targets[i:i + split_size],
        "ligand": ligand[i:i + split_size],
    }
    # 부분적으로 저장
    with open(f"{save_path}pdbbind_pka_data_part_{i//split_size}.pkl", "wb") as f:
        pickle.dump(data_chunk, f)

    print(f"Saved: pdbbind_data_part_{i//split_size}.pkl")

print("데이터 분할 저장 완료")
############################################################################################################################################ㅍ
# %%
import pickle
import os

save_path = "/home/bioscience/dev/DeepInteract_Recomb/features/splits/"
final_save_path = "/home/bioscience/dev/DeepInteract_Recomb/features/all/pdbbind_data.pkl"

# 합칠 파일 리스트 가져오기
pkl_files = sorted([f for f in os.listdir(save_path) if f.endswith(".pkl")])

final_data = {
    "one_hot_nodes": [],
    "adjs": [],
    "targets": [],
    "ligand": [],
}

total_len_nodes = 0
total_len_adjs = 0
total_len_targets = 0
total_len_ligand = 0
# %%
# 각 파일을 읽어서 데이터를 합치고, 각 파일의 len을 출력
for pkl_file in pkl_files:
    with open(os.path.join(save_path, pkl_file), "rb") as f:
        data_chunk = pickle.load(f)

        len_nodes = len(data_chunk["one_hot_nodes"])
        len_adjs = len(data_chunk["adjs"])
        len_targets = len(data_chunk["targets"])
        len_ligand = len(data_chunk["ligand"])

        # 각 데이터의 길이를 더함
        total_len_nodes += len_nodes
        total_len_adjs += len_adjs
        total_len_targets += len_targets
        total_len_ligand += len_ligand

        # 데이터 병합
        final_data["one_hot_nodes"].extend(data_chunk["one_hot_nodes"])
        final_data["adjs"].extend(data_chunk["adjs"])
        final_data["targets"].extend(data_chunk["targets"])
        final_data["ligand"].extend(data_chunk["ligand"])

    # 각 파일의 len을 출력
    print(f"Loaded: {pkl_file}, len(one_hot_nodes)={len_nodes}, len(adjs)={len_adjs}, len(targets)={len_targets}, len(ligand)={len_ligand}")

# %%
# 최종 데이터를 하나의 pkl 파일로 저장
with open(final_save_path, "wb") as f:
    pickle.dump(final_data, f)

# 최종 길이 출력
print("\n총 데이터 길이:")
print(f"Total len(one_hot_nodes)={total_len_nodes}")
print(f"Total len(adjs)={total_len_adjs}")
print(f"Total len(targets)={total_len_targets}")
print(f"Total len(ligand)={total_len_ligand}")

print("데이터 병합 및 저장 완료")