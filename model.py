"""Backward-compatible import for the new single-head PSM."""

from ea_model.psm import ParetoSetMLP


class ParetoSetModel(ParetoSetMLP):
    """Simple MLP mapping PF locations directly to decision vectors."""

    def __init__(self, n_dim: int, n_obj: int, hidden_size: int = 128, num_layers: int = 3):
        super().__init__(n_obj, n_dim, hidden_size, num_layers)
