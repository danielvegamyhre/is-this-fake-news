import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correction averaging."""

    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

