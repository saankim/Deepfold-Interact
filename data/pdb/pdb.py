# %%
from utils.dataset import MyDataset
import pickle
import os

def load_data():
    save_path = "/home/bioscience/dev/DeepInteract_Recomb/features/splits/"
    # 합칠 파일 리스트 가져오기
    pkl_files = sorted([f for f in os.listdir(save_path) if f.endswith(".pkl")])

    final_data = {
        "one_hot_nodes": [],
        "adjs": [],
        "targets": [],
        "ligand": [],
    }
    # %%
    # 각 파일을 읽어서 데이터를 합치고, 각 파일의 len을 출력
    for pkl_file in pkl_files:
        with open(os.path.join(save_path, pkl_file), "rb") as f:
            data_chunk = pickle.load(f)

            # 데이터 병합
            final_data["one_hot_nodes"].extend(data_chunk["one_hot_nodes"])
            final_data["adjs"].extend(data_chunk["adjs"])
            final_data["targets"].extend(data_chunk["targets"])
            final_data["ligand"].extend(data_chunk["ligand"])

    return final_data

class PDBBindDataset(MyDataset):
    def __init__(
        self,
        #path="./dataset/qm9/qm9_data.pkl",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
    ):
        #data = pickle.load(open(path, "rb"))
        data = load_data()
        data["one_hot_nodes"] = [nf.squeeze(0) for nf in data["one_hot_nodes"]]

        super().__init__(
            data["one_hot_nodes"],
            data["adjs"],
            data["targets"],
            data["ligand"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size,
        )
