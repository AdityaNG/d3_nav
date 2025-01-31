import torch
from torch.nn import functional as F


def gumbel_logits_and_indices(logits, temperature=0.1, noise=True):
    """
    Performs Gumbel-Softmax sampling on the logits and extracts the embeddings.

    Args:
    - logits (torch.Tensor): The input logits of shape (B, 128, 1024).
    - embedding_matrix (torch.Tensor): The embedding matrix of shape
        (1024, embedding_dim).
    - temperature (float): The temperature for the Gumbel-Softmax sampling.

    Returns:
    - embedding_features (torch.Tensor): The extracted embedding features.
    """

    # Gumbel-Softmax sampling
    gumbel_noise = sample_gumbel(logits.size()).to(logits.device)
    if noise:
        gumbel_logits = logits + gumbel_noise
    else:
        gumbel_logits = logits
    gumbel_logits = F.softmax(gumbel_logits / temperature, dim=-1)
    _, gumbel_indices = gumbel_logits.max(dim=-1)

    return gumbel_logits, gumbel_indices


def gumbel_sample(gumbel_logits, embedding_matrix):
    embedding_features = torch.matmul(gumbel_logits, embedding_matrix)
    return embedding_features


def gumbel_softmax_sample_and_embed(
    logits, embedding_matrix, temperature=0.1, noise=True
):
    """
    Performs Gumbel-Softmax sampling on the logits and extracts the embeddings.

    Args:
    - logits (torch.Tensor): The input logits of shape (B, 128, 1024).
    - embedding_matrix (torch.Tensor): The embedding matrix of shape
        (1024, embedding_dim).
    - temperature (float): The temperature for the Gumbel-Softmax sampling.

    Returns:
    - embedding_features (torch.Tensor): The extracted embedding features.
    """

    # Gumbel-Softmax sampling
    gumbel_noise = sample_gumbel(logits.size()).to(logits.device)
    if noise:
        gumbel_logits = logits + gumbel_noise
    else:
        gumbel_logits = logits
    y_soft = F.softmax(gumbel_logits / temperature, dim=-1)
    _, y_hard_indices = y_soft.max(dim=-1)

    # Extract embedding features
    embedding_features = torch.matmul(y_soft, embedding_matrix)

    return embedding_features


def sample_gumbel(shape, eps=1e-20):
    """
    Samples from the Gumbel distribution.

    Args:
    - shape (tuple): The shape of the tensor to sample.
    - eps (float): A small constant to prevent numerical instability.

    Returns:
    - gumbel_samples (torch.Tensor): Samples from the Gumbel distribution.
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def multinomial_torch(prob_matrix):
    prob_matrix /= prob_matrix.sum(dim=2, keepdim=True)
    s = prob_matrix.cumsum(dim=2)
    r = torch.rand(
        (prob_matrix.shape[0], prob_matrix.shape[1]), device=prob_matrix.device
    )
    k = (s < r.unsqueeze(2)).sum(dim=2)
    return k
