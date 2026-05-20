from functools import partial
import torch
from transformers import LogitsProcessor, LogitsProcessorList

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

class VLALogitsProcessor(LogitsProcessor):  
    def __init__(self, vocab_size: int, embedding_matrix: torch.FloatTensor, similarity_scheme: str = "cosine", 
                 input_embeddings: torch.LongTensor = None, special_tokens: list[int] = None, split_x: int = 2, 
                 atten_delta = 0.02, sct_radio =0.025, alpha = 0.4, trans_type = None, attn_threshold=0.2) -> None:
        self.special_tokens = special_tokens
        self.embedding_matrix = embedding_matrix
        self.hash_key = 15485863
        self.vocab_size = vocab_size
        self.gamma = round(1 / split_x, 2)
        self.prefix_length = 1
        self.delta = 3.0
        self.water_topp = 0.98
        self.similarity_scheme = SimilarityScheme(similarity_scheme)
        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, device='cuda', generator=self.rng)
        self.sim_rng = torch.Generator(device='cpu') 
        self.msg = None
        self.list_ids = None
        self.similarity_list = None
        self.M_ids = None
        self.cls = None
        self.entropy_max = None
        self.entropy_current = None
        self.attention_current = None
        self.similarity_matrix = None
        self.b_list = []
        if input_embeddings is not None: 
            self.refresh_msg(input_embeddings)
        self.attention_scores = None
        self.watermark_switch = False
        self.atten_delta = atten_delta
        self.sct_radio = sct_radio
        self.imgs_pos = None
        self.max_gamma = -1
        self.alpha = alpha
        self.hidden_states = None    
        self.trans_type = trans_type
        self.attn_threshold = attn_threshold
        self.attn_entropy_current = None
