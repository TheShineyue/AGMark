import torch
class SimilarityScheme():
    def __init__(self, similarity_scheme: str) -> None:
        self.similarity_scheme = similarity_scheme
        self.similarity_scheme_map = {
            "cosine": self._cosine_similarity,
            "dot_product": self._dot_product_similarity,
        }
    def _cosine_similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        if m1.dim() == 1:
            m1 = m1.unsqueeze(0)
        if m2.dim() == 1:
            m2 = m2.unsqueeze(0)
        # assert not torch.isnan(m1).any(), "m1 contains NaN"
        # assert not torch.isnan(m2).any(), "m2 contains NaN"
        m1_norm = torch.norm(m1, dim=1, keepdim=True).clamp_min(1e-12)
        m2_norm = torch.norm(m2, dim=1, keepdim=True).clamp_min(1e-12)
        dot_product = torch.matmul(m1, m2.T)
        norm_product = m1_norm * m2_norm.T
        return dot_product / norm_product
    def _dot_product_similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        if m1.dim() == 1:
            m1 = m1.unsqueeze(0)
        if m2.dim() == 1:
            m2 = m2.unsqueeze(0)
        return torch.matmul(m1, m2.T)
    def similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        # assert m1 is not None, "m1 should not be None"
        # assert m2 is not None, "m2 should not be None"
        target_device = m2.device
        target_dtype = torch.float32
        if m1.device != target_device:
            m1 = m1.to(target_device)
        if m2.device != target_device:
            m2 = m2.to(target_device)
        if m1.dtype != target_dtype:
            m1 = m1.to(dtype=target_dtype)
        if m2.dtype != target_dtype:
            m2 = m2.to(dtype=target_dtype)
        return self.similarity_scheme_map[self.similarity_scheme](m1, m2)
def z_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-12)
