#%%
import re
import numpy as np

# Function to extract Kd, Ki, or IC50 value and convert it to molar value, including comparison operators and small units
def extract_kd_ki_ic50(value_str):
    # Match patterns for Kd, Ki, IC50 including comparison operators and range symbols
    match = re.search(r'([Kk][id]=|IC50=|Ki~|Kd>|IC50~|IC50>|Ki<=|Kd<=|Ki>=|Kd>=|Kd~|Ki~|IC50<|Ki>|Ki<|Kd<|IC50>=|)([0-9\.]+)([a-zA-Z]+)', value_str)
    if match:
        value = float(match.group(2))
        unit = match.group(3).lower()
        
        # Convert to molar (including small units like fM, pM, etc.)
        unit_conversion = {
            'mm': 1e-3,  # millimolar to molar
            'um': 1e-6,  # micromolar to molar
            'nm': 1e-9,  # nanomolar to molar
            'pm': 1e-12, # picomolar to molar
            'fm': 1e-15  # femtomolar to molar
        }
        if unit in unit_conversion:
            return value * unit_conversion[unit]
    
    return None  # If not matched or invalid unit

# File path to the dataset
file_path = "/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/"

# Reading the raw data
#with open(file_path + "INDEX_general_PL_data.2020", 'r') as file:
with open(file_path + "INDEX_refined_data.2020", 'r') as file:
    raw_data = file.readlines()

# Extract PDB Code and Kd/Ki/IC50, then convert to -log(Kd/Ki/IC50)
data_dict = {}
missing_data = []  # List to store missing entries
for line in raw_data[6:]:  # Start after the header lines
    # Split the line by whitespace
    parts = line.split()
    
    if len(parts) >= 5:
        pdb_code = parts[0]
        kd_ki_ic50_str = parts[4]  # This is the string that contains Kd, Ki, or IC50
        kd_ki_ic50_value = extract_kd_ki_ic50(kd_ki_ic50_str)
        
        if kd_ki_ic50_value:
            # Calculate -log(Kd/Ki/IC50)
            log_kd_ki_ic50 = -np.log10(kd_ki_ic50_value)
            data_dict[pdb_code] = log_kd_ki_ic50
        else:
            missing_data.append(line)  # If no Kd/Ki/IC50 found, add to missing_data

# Show the length of the extracted data and missing data
print(f"Total extracted entries: {len(data_dict)}")
print(f"Total missing entries: {len(missing_data)}")

# Optionally, save the missing data for analysis
with open(file_path + 'missing_data_log_final.txt', 'w') as missing_file:
    for entry in missing_data:
        missing_file.write(entry + '\n')

# Save the result as a JSON file
import json

#with open(file_path + 'pdb_log_general_final.json', 'w') as json_file:
with open(file_path + 'pdb_log_refined_final.json', 'w') as json_file:
    json.dump(data_dict, json_file)

#%%
import json

# 파일 경로 설정
general_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/pdb_log_general_final.json'
refined_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/pdb_log_refined_final.json'

# 두 개의 JSON 파일 불러오기
with open(general_file_path, 'r') as general_file:
    general_data = json.load(general_file)

with open(refined_file_path, 'r') as refined_file:
    refined_data = json.load(refined_file)

# 겹치는 key 찾기
common_keys = set(general_data.keys()).intersection(refined_data.keys())

# 동일한 key를 가지고 있지만 value가 다른 항목을 찾음
different_values = {}
for key in common_keys:
    if general_data[key] != refined_data[key]:
        different_values[key] = {
            'general_value': general_data[key],
            'refined_value': refined_data[key]
        }

# 겹치는 key 개수 출력
print(f"Total common keys: {len(common_keys)}")

# 결과 출력: value가 다른 key 개수
print(f"Total keys with different values: {len(different_values)}")

# 값이 다른 항목 출력
for key, values in different_values.items():
    print(f"Key: {key}")
    print(f"General Value: {values['general_value']}")
    print(f"Refined Value: {values['refined_value']}")
    print()

# %%
import json
import torch

# Load the JSON file (data_dict)
json_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/pdb_log_general_final.json'
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

# Load the name.pth file
path_name = "/home/bioscience/dev/DeepInteract_Recomb/features/all/name.pth"
with open(path_name, 'rb') as file:
    name = torch.load(file)

# Print the lengths of both datasets
print(f"Length of data_dict: {len(data_dict)}")
print(f"Length of name: {len(name)}")

# Find the non-matching PDB codes
non_matching_indices = []
non_matching_count = 0

# Iterate through 'name' and compare with data_dict keys
for i, pdb_name in enumerate(name):
    if pdb_name not in data_dict:
        non_matching_count += 1
        non_matching_indices.append(i)

# Print the count of non-matching entries and their indices
print(f"Number of non-matching entries: {non_matching_count}")
print(f"Indices of non-matching entries: {non_matching_indices}")
# %%
# %%
import json
import torch

# Load the JSON file (data_dict)
json_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/pdb_log_general_final.json'
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

# Load the name.pth file
path_name = "/home/bioscience/dev/DeepInteract/features/all/name.pth"
with open(path_name, 'rb') as file:
    name = torch.load(file)

# Print the lengths of both datasets
print(f"Length of data_dict: {len(data_dict)}")
print(f"Length of name: {len(name)}")

# Create a list to store matching PDB entries in the same order as 'name.pth'
name_filtered_ordered = [data_dict[pdb_name] for pdb_name in name if pdb_name in data_dict]

# Verify if the order is the same
order_is_same = all(name[i] in data_dict and data_dict[name[i]] == name_filtered_ordered[i] for i in range(len(name_filtered_ordered)))

# Print the result
if order_is_same:
    print("The order of data_dict values matches the order of name.pth.")
else:
    print("The order of data_dict values does NOT match the order of name.pth.")

# Optionally, you can save the filtered and ordered data to a new JSON file
output_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/name_filtered_ordered_dict.json'
with open(output_file_path, 'w') as output_file:
    json.dump({name[i]: name_filtered_ordered[i] for i in range(len(name_filtered_ordered))}, output_file, indent=4)

print(f"Filtered and ordered dictionary saved to {output_file_path}")

# %%
import json
import torch

# Load the filtered and ordered dictionary from JSON
filtered_ordered_file_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/name_filtered_ordered_dict.json'
with open(filtered_ordered_file_path, 'r') as json_file:
    name_filtered_ordered_dict = json.load(json_file)

# Extract only the values (without the keys)
values_list = list(name_filtered_ordered_dict.values())
#%%
# Convert the list of values to a tensor
values_tensor = torch.tensor(values_list)

# Print the shape of the tensor to verify
print(f"Shape of the tensor: {values_tensor.shape}")
#%%
# Optionally, you can save the tensor to a .pth file
output_tensor_path = '/home/bioscience/dev/DeepInteract_Recomb/Recomb/PDB/values_tensor.pth'
torch.save(values_list, output_tensor_path)

print(f"Tensor saved to {output_tensor_path}")

# %%
