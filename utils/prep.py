# %%
import torch
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# from Bio import pairwise2
from rdkit.Chem import MolFromMol2File, MolToSmiles
import ankh
import Bio.PDB
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem


# %%
class PretrainedSequenceEmbedding:
    def __init__(self):
        self.model, self.tokenizer = ankh.load_large_model()
        self.model.eval()

    def __call__(self, sequence):
        protein_sequences = []
        protein_sequences.append(sequence)
        protein_sequences = [list(seq) for seq in protein_sequences]
        outputs = self.tokenizer.batch_encode_plus(
            protein_sequences,
            add_special_tokens=False,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = self.model(
                input_ids=outputs[
                    "input_ids"
                ],  # Move input tensors to the specified device
                attention_mask=outputs["attention_mask"],
            )
        return embeddings.last_hidden_state


class PretrainedMolecularEmbedding:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )

    def operate_list(self, *smiles):
        smiles_list = list(smiles)
        input_list = self.tokenizer(
            smiles_list,
            padding=True,
            # truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output_list = self.model(**input_list)
        return output_list.pooler_output

    def __call__(self, smiles):
        return self.operate_list(smiles)[0]


def mol2_to_smiles(mol2_filepath):
    # Read the MOL2 file
    mol = MolFromMol2File(mol2_filepath)
    if mol is None:
        raise ValueError("Invalid MOL2 file. Could not parse molecule.")
    # Convert to SMILES
    smiles = MolToSmiles(mol)
    return smiles


def make_sequences(pdb_location):
    # Create a parser with permissive set to True to ignore minor issues
    parser = PDBParser(PERMISSIVE=False, QUIET=True)
    # Parse the structure from file
    structure = parser.get_structure("PDB_structure", pdb_location)
    # This will store the sequences of each chain
    sequences = []
    # Loop over each model (usually there is only one model in a PDB file)
    for model in structure:
        # Loop over each chain in the model
        for chain in model:
            # Collect each residue in the chain if it is an amino acid
            sequence = ""
            for residue in chain:
                # Check if the residue name is in the standard amino acids
                if is_aa(residue):
                    # Append the one-letter code of the amino acid to the sequence
                    sequence += seq1(residue.get_resname())
            if sequence:  # If there's a valid sequence, add it to the list
                sequences.append(sequence)
    return sequences


def make_a_sequence(pdb_location):
    return "".join(make_sequences(pdb_location))


def make_coordinates(protein_pdb):
    parser = PDBParser(PERMISSIVE=False, QUIET=True)
    structure = parser.get_structure("Protein", protein_pdb)
    model = structure[0]  # Assume the structure contains only one model.
    ca_atoms = np.array(
        [residue["CA"].coord for residue in model.get_residues() if "CA" in residue]
    )
    return torch.tensor(ca_atoms)


def make_sequence_and_coordinates(protein_pdb):
    # Create a parser# Create a parser object
    parser = Bio.PDB.PDBParser(QUIET=True)
    # Parse the structure from the PDB file
    structure = parser.get_structure("protein", protein_pdb)
    sequences = []
    # Loop over each model (usually there is only one model in a PDB file)
    for model in structure:
        # Loop over each chain in the model
        for chain in model:
            # Collect each residue in the chain if it is an amino acid
            sequence = ""
            for residue in chain:
                # Check if the residue name is in the standard amino acids
                if is_aa(residue):
                    # Append the one-letter code of the amino acid to the sequence
                    sequence += seq1(residue.get_resname())
            if sequence:  # If there's a valid sequence, add it to the list
                sequences.append(sequence)
    merged_sequence = "".join(sequences)
    # Initialize variables
    residue_coordinates = []
    seq_index = 0
    # Iterate through each model in the structure (can handle multimers)
    for model in structure:
        for chain in model:
            for residue in chain:
                if seq_index == len(merged_sequence):
                    break
                # Get the one-letter code of the residue
                residue_name = Bio.SeqUtils.seq1(residue.get_resname())
                residue_name = "X" if residue_name == "" else residue_name
                # Check if this residue matches the next in the merged sequence
                if residue_name == merged_sequence[seq_index]:
                    # Extract CA atom coordinate
                    try:
                        ca_atom = residue["CA"]
                        ca_coord = ca_atom.get_coord()
                        residue_coordinates.append(ca_coord)
                    except KeyError:
                        # If CA atom is missing, leave a None entry to be interpolated later
                        residue_coordinates.append(None)
                    seq_index += 1
                    # Stop if the entire sequence has been processed
    # Interpolate missing CA coordinates
    for i, coord in enumerate(residue_coordinates):
        if coord is None:
            # Find the next available coordinate
            start = i
            while i < len(residue_coordinates) and residue_coordinates[i] is None:
                i += 1
            end = i
            # Get the coordinates before and after the missing ones
            prev_coord = residue_coordinates[start - 1]
            next_coord = (
                residue_coordinates[end]
                if end < len(residue_coordinates)
                else prev_coord
            )
            # break if any coordinate is missing
            if prev_coord is None or next_coord is None:
                break
            # Interpolate the coordinates linearly
            for j in range(start, end):
                interpolated_coord = prev_coord + (next_coord - prev_coord) * (
                    j - start + 1
                ) / (end - start + 1)
                residue_coordinates[j] = interpolated_coord
    # Delete None from residue_coordinates
    # and delete the corresponding residues from the merged_sequence
    # Delete None from residue_coordinates
    merged_sequence = [
        merged_sequence[i]
        for i in range(len(residue_coordinates))
        if residue_coordinates[i] is not None
    ]
    residue_coordinates = [coord for coord in residue_coordinates if coord is not None]
    assert len(residue_coordinates) == len(
        merged_sequence
    ), f"Length mismatch, {len(residue_coordinates)} != {len(merged_sequence)}"
    assert len(residue_coordinates) != 0, "No residue coordinates"
    return (merged_sequence, torch.tensor(np.stack(residue_coordinates, axis=0)))


def make_mask_fast(protein_pdb, distance_threshold=10.0, metric="euclidean"):
    parser = PDBParser(PERMISSIVE=False, QUIET=True)
    structure = parser.get_structure("Protein", protein_pdb)
    model = structure[0]  # Assume the structure contains only one model.
    # Collect all residue CA atoms (assuming only standard amino acids)
    ca_atoms = np.array(
        [residue["CA"].coord for residue in model.get_residues() if "CA" in residue]
    )
    diff = ca_atoms[:, np.newaxis, :] - ca_atoms[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    # Create the mask based on the distance threshold
    mask = np.where(distances <= distance_threshold, 0, float("-inf"))
    # Convert the mask to a PyTorch tensor and add a batch dimension
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    return mask_tensor


# %%
def get_ligand_center_of_mass(mol2_file):
    # Read the mol2 file and convert it to an RDKit molecule
    mol = MolFromMol2File(mol2_file)
    if mol is None:
        raise ValueError("Could not parse the mol2 file.")
    atom_weights = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    positions = np.array(
        [list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    total_mass = atom_weights.sum()
    return np.dot(atom_weights, positions) / total_mass


def get_ligand_center_of_mass2(mol2_file):
    # Atomic masses for common elements (in atomic mass units)
    atomic_masses = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
        "Br": 79.904,
        "F": 18.998,
        "I": 126.90,
        "Ca": 40.078,
        # Add more elements as needed
    }

    total_mass = 0.0
    center_of_mass = [0.0, 0.0, 0.0]

    with open(mol2_file, "r") as file:
        lines = file.readlines()

    atom_section = False

    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            atom_section = True
            continue
        elif line.startswith("@<TRIPOS>"):
            atom_section = False

        if atom_section:
            parts = line.split()
            atom_name = parts[
                5
            ]  # The atom name or element is typically in the 6th column

            # Handle cases where the atom name might be a single or double letter
            if atom_name in atomic_masses:
                mass = atomic_masses[atom_name]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                total_mass += mass
                center_of_mass[0] += mass * x
                center_of_mass[1] += mass * y
                center_of_mass[2] += mass * z

    if total_mass == 0:
        fallback = get_ligand_center_of_mass(mol2_file)
        if np.isnan(fallback).any():
            raise ValueError("Total mass is zero, check the MOL2 file for valid atoms.")
        return fallback

    center_of_mass[0] /= total_mass
    center_of_mass[1] /= total_mass
    center_of_mass[2] /= total_mass

    return torch.tensor(center_of_mass)


def calculate_distances(tup):
    residue_coords, ligand_com = tup
    distances = []
    for coord in residue_coords:
        distance = np.linalg.norm(coord - ligand_com)
        distances.append(distance)
    return torch.tensor(distances)
