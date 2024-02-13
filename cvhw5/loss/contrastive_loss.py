import torch
import torch.nn.functional as F


def get_contrastive_loss(z: torch.Tensor):
    """
    Args:
        z(torch.Tensor): (N, D)
    """
    device = z.device

    assert z.shape[0] % 2 == 0
    N = z.shape[0] // 2

    z = F.normalize(z, dim=1)
    sim_matrix = torch.mm(z, z.T)
    sim_matrix = torch.exp(sim_matrix)

    # remove identity similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, 0.0)
    sim_matrix /= torch.sum(sim_matrix, dim=1).view(2 * N, 1)

    mask = torch.zeros((2 * N, 2 * N), dtype=torch.bool, device=device)
    d = (torch.arange(1, 2 * N, device=device) % 2).bool()

    mask = torch.diagonal_scatter(mask, d, 1)
    mask = torch.diagonal_scatter(mask, d, -1)

    items = sim_matrix[mask]

    items = -torch.log(items)

    return items.mean()
