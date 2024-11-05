# %%
import torch

def conan(tensor, msg):
    if torch.isnan(tensor).any():
        raise ValueError(f"The tensor {msg} has NaN values.")

def none_for_nan(tensor):
    try:
        if not torch.isnan(tensor).any():
            return tensor
        return None
    except:
        return None

def clean_lists(*lists):
    # Check if all lists have the same length
    length = len(lists[0])
    for lst in lists:
        if len(lst) != length:
            raise ValueError("All lists must have the same length")
    # Identify indices where any of the lists has a None value
    indices_to_remove = [
        i for i in range(length) if any(lst[i] is None for lst in lists)
    ]
    # Remove the identified elements from all lists
    cleaned_lists = [
        [lst[i] for i in range(length) if i not in indices_to_remove] for lst in lists
    ]
    return cleaned_lists
