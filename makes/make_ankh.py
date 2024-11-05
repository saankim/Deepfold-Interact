# %%
import torch
import ankh

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

# %%
if __name__ == "__main__":
    seq = "XMDXRTCEERPAEDGSDEEDPDSMEAPTRIRDTPEDIVLEAPASGLAFHPARDLLAAGDVDGDVFVFSYSCQEGETKELWSSGHHLKACRAVAFSEDGQKLITVSKDKAIHVLDVEQGQLERRVSKAHGAPINSLLLVDENVLATGDDTGGICLWDQRKEGPLMDMRQHEEYIADMALDPAKKLLLTASGDGCLGIFNIKRRRFELLSEPQSGDLTSVTLMKWGKKVACGSSEGTIYLFNWNGFGATSDRFALRAESIDCMVPVTESLLCTGSTDGVIRAVNILPNRVVGSVGQHTGEPVEELALSHCGRFLASSGHDQRLKFWDMAQLRAVVVDDYRRRKKKGGPLRALSSKTWSTDDFFAGLREEGEDSMAQEEKEETGDDSD"
    embedder = PretrainedSequenceEmbedding()
    e = embedder(seq)
    print(e.shape)
