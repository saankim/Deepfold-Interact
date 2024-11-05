# %% imports
import os
import torch
from IPython.display import display
from utils.pipe import Pipe
from utils.prep import (
    PretrainedMolecularEmbedding,
    PretrainedSequenceEmbedding,
    mol2_to_smiles,
    make_edge_features,
    make_a_sequence,
    make_coordinates,
    make_sequence_and_coordinates,
    get_ligand_center_of_mass2,
    calculate_distances,
)
from utils.cleansing import (
    clean_lists,
    none_for_nan,
)
import pickle
import logging

# 파일 사용법 예시
# ulimit -n 320000 && nohup python pre.py | tee output_16000-19300.log & disown

logging.basicConfig(level=logging.INFO)
starting_number = 0
ending_number = 4000

nproc = os.cpu_count() - 2

MERGED_DIR = "/home/bioscience/datasets/deepinteract-dataset/merged/"
PROTEIN_FILE_FORMAT = MERGED_DIR + "proteins/{}_protein.pdb"
LIGAND_FILE_FORMAT = MERGED_DIR + "ligands/{}_ligand.mol2"
# POCKET_FILE_FORMAT = MERGED_DIR + "pockets/{}_pocket.pdb"
# DSSP_FILE_FORMAT = MERGED_DIR + "dssps/{}_protein.dssp"

names = [
    file[:4]
    for file in os.listdir(
        "/home/bioscience/datasets/deepinteract-dataset/merged/proteins.atoms"
    )
    if file.endswith("_protein.pdb")
][starting_number:ending_number]

# %%
proteins = [PROTEIN_FILE_FORMAT.format(name) for name in names]
ligands = [LIGAND_FILE_FORMAT.format(name) for name in names]
# pockets = [POCKET_FILE_FORMAT.format(name) for name in names]
# dssps = [DSSP_FILE_FORMAT.format(name) for name in names]
logging.info(f"{len(names)} data entries to process")

# coordinates
p_coor = Pipe(proteins, make_sequence_and_coordinates)
seq_and_coord = p_coor.run_multiproc(cores=nproc)
sequences, coordinates = zip(*seq_and_coord)
logging.info("DONE: sequence and coordinates")

# regression target
p_lcom = Pipe(ligands, get_ligand_center_of_mass2)
ligand_coms = p_lcom.run_multiproc(cores=nproc)
logging.info("DONE: ligand center of mass")
p_regt = Pipe(list(zip(coordinates, ligand_coms)), calculate_distances)
list_targ = p_regt.run_multiproc(cores=nproc)
logging.info("DONE: regression target")


# sequences
# p_sequ = Pipe(proteins, make_a_sequence)
# sequences = p_sequ.run_multiproc(cores=nproc)
# logging.info("DONE: sequences")

# smiles
p_smil = Pipe(ligands, mol2_to_smiles)
smiles = p_smil.run_multiproc(cores=nproc)
logging.info("DONE: smiles")

# %%
# GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_device(DEVICE)
logging.info(f"Using device: {DEVICE}")

# molformer
molformer = PretrainedMolecularEmbedding()
p_mol = Pipe(smiles, molformer)
list_mole = p_mol.run_each()
logging.info("DONE: molformer")

# ankh
ankh = PretrainedSequenceEmbedding()
p_ankh = Pipe(sequences, ankh)
list_node = p_ankh.run_each()
logging.info("DONE: ankh")


# find and where is nan
def log_nan(l):
    for i, x in enumerate(l):
        if none_for_nan(x) is None:
            logging.info(f"node: {i}")
    return None

# log number of data points
logging.info(f"names: {len(names)}")
logging.info(f"seq: {len(sequences)}")
logging.info(f"coor: {len(coordinates)}")
logging.info(f"node: {len(list_node)}")
logging.info(f"mole: {len(list_mole)}")
logging.info(f"targ: {len(list_targ)}")

# none for nan
nonfornan_list_coor = list(map(none_for_nan, coordinates))
nonfornan_list_node = list(map(none_for_nan, list_node))
nonfornan_list_mole = list(map(none_for_nan, list_mole))
nonfornan_list_targ = list(map(none_for_nan, list_targ))

# clean lists
(
    cleaned_list_names,
    cleaned_list_seq,
    cleaned_list_coor,
    cleaned_list_node,
    cleaned_list_mole,
    cleaned_list_targ,
) = clean_lists(
    names,
    sequences,
    nonfornan_list_coor,
    nonfornan_list_node,
    nonfornan_list_mole,
    nonfornan_list_targ,
)
logging.info(f"names_cleaned: {len(cleaned_list_names)}")
logging.info(f"seq_cleaned: {len(cleaned_list_seq)}")
logging.info(f"coor_cleaned: {len(cleaned_list_coor)}")
logging.info(f"node_cleaned: {len(cleaned_list_node)}")
logging.info(f"mole_cleaned: {len(cleaned_list_mole)}")
logging.info(f"targ_cleaned: {len(cleaned_list_targ)}")
logging.info("DONE: cleansing")

os.chdir("/home/bioscience/dev/DeepInteract_Recomb/Recomb/features")
with open(f"name_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_names, file)
with open(f"seq_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_seq, file)
with open(f"coor_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_coor, file)
with open(f"node_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_node, file)
with open(f"mol_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_mole, file)
with open(f"regr_{starting_number}-{ending_number}.pkl", "wb") as file:
    pickle.dump(cleaned_list_targ, file)
logging.info("DONE: saved to disk")
