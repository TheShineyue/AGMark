# from functools import partial
from functools import partial
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from math import sqrt, exp
from markllm.watermark.base import BaseWatermark
# import torch.nn.functional as F
import numpy as np
import time
import math
import PIL
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

    def set_attention_scores(self, attention_scores: torch.LongTensor) -> None:
        self.attention_scores = attention_scores[0][self.imgs_pos[0] + 1 : self.imgs_pos[1]]

    def refresh_msg(self, input_embeddings: torch.LongTensor) -> None:
        msg = input_embeddings.squeeze(0)
        self.msg = msg
        self.refresh_similarity_list()

    def refresh_similarity_list(self) -> None:
        """刷新相似度列表"""
        self._calculate_similarity_list()

    def runing_refresh_similarity_list(self) -> None:
        self._runing_calculate_similarity_list()

    def _calculate_similarity_list(self) -> float:
        """计算相似度列表"""
        msg = self.msg
        self.similarity_matrix = self.similarity_scheme.similarity(msg, self.embedding_matrix).to("cuda") 
        raw_v_list = self._calculate_attention_relevance()
        v_list = z_norm(raw_v_list) 
        raw_t_list = self._calculate_context_relevance()
        t_list = z_norm(raw_t_list)
        beta = self.atten_delta
        similarity_list = beta * v_list + (1- beta) * t_list
        similarity_list = self._normalize(similarity_list)
        self.similarity_list = similarity_list

    def _calculate_attention_relevance(self) -> torch.LongTensor:
        weighted_sum = self.attention_scores.unsqueeze(0).T * self.similarity_matrix
        CAS_similarity_list = weighted_sum.sum(dim=0)
        CAS_similarity_list = CAS_similarity_list - CAS_similarity_list.max() 
        CAS_similarity_list = torch.softmax(CAS_similarity_list, dim=0)
        return CAS_similarity_list
    def _calculate_context_relevance(self) -> torch.LongTensor:
        weighted_sum = self.similarity_scheme.similarity(self.hidden_states, self.embedding_matrix).to("cuda") 
        CAS_similarity_list = weighted_sum.sum(dim=0)
        CAS_similarity_list = CAS_similarity_list - CAS_similarity_list.max() 
        CAS_similarity_list = torch.softmax(CAS_similarity_list, dim=0)
        return CAS_similarity_list
    def _runing_calculate_similarity_list(self) -> float:
        raw_v_list = self._calculate_attention_relevance()
        v_list = z_norm(raw_v_list)
        raw_t_list = self._calculate_context_relevance()
        t_list = z_norm(raw_t_list)
        beta = self.atten_delta
        similarity_list = beta * v_list + (1- beta) * t_list
        similarity_list = self._normalize(similarity_list)
        self.similarity_list = similarity_list
    
    def top_p_count_from_scores(self, scores: torch.Tensor, top_p: float, temperature: float = 1.0):
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        if temperature != 1.0:
            sorted_scores = sorted_scores / float(temperature)
        sorted_probs = torch.softmax(sorted_scores, dim=0)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        k_idx = int(torch.searchsorted(cum_probs, torch.tensor(top_p, device=cum_probs.device)).item())
        k = k_idx + 1
        kept_ids = sorted_ids[:k]
        kept_probs = sorted_probs[:k]
        return k, kept_ids, kept_probs, cum_probs
    
    def _onlyimg_similarity_normal_grouping(self, similarity_list: torch.LongTensor):
        """随机扰动项lambda"""
        similarity_list = self.similarity_list
        k, kept_ids, kept_probs, cum_probs = self.top_p_count_from_scores(similarity_list, self.water_topp)
        current_gamma = self.sct_radio * (k / self.vocab_size)  *(1.0 - self.entropy_current / self.entropy_max)
        sorted_indices = torch.argsort(similarity_list, descending=True)
        list_size = int(self.vocab_size * current_gamma)
        M_ids = sorted_indices[:list_size].tolist()
        random_permutation_full = torch.randperm(self.vocab_size, generator=self.sim_rng).tolist()
        M_ids_set = set(M_ids)
        random_permutation = [i for i in random_permutation_full if i not in M_ids_set]
        split_index = int(self.vocab_size * 0.5 - len(M_ids))
        G_ids = random_permutation[:split_index]
        R_ids = random_permutation[split_index:]
        list_ids = [[] for _ in range(2)]
        list_ids[0] = M_ids + G_ids  # 绿色词表
        list_ids[1] = R_ids          # 红色词表
        return list_ids
    
    def similarity_grouping(self, similarity_list: torch.LongTensor, position: int) -> list[list[int]]:
        """根据SCT随机化相似度列表"""
        self.sim_rng.manual_seed(position * self.hash_key)
        return self._onlyimg_similarity_normal_grouping(similarity_list)

    def get_list_ids(self, position: int) -> list[list[int]]:
        """根据msg和embedding_matrix计算相似度列表, 并根据相似度列表进行分割词汇表并获取绿色词表"""
        similarity_list = self.similarity_list
        if similarity_list.shape[0] > self.vocab_size:
            similarity_list = similarity_list[:self.vocab_size]
        
        list_ids = self.similarity_grouping(similarity_list, position)
        return list_ids
    
    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens and output the biased scores."""
        inverted_mask = ~greenlist_mask.to("cuda")
        scores.add_(inverted_mask * (-greenlist_bias))       
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""

        if not self.watermark_switch:
            return scores
        if input_ids.shape[-1] < self.prefix_length:
            return scores
        position = self._f(input_ids[0])

        squeeze_scores = scores.squeeze().to(scores.device)
        squeeze_scores = torch.nan_to_num(
            squeeze_scores,
            nan=-10.0,
            posinf=None,
            neginf=-10.0
        ).to(dtype=scores.dtype)
        self.entropy_max = math.log(self.vocab_size)
        self.entropy_current = torch.distributions.Categorical(logits=squeeze_scores).entropy().item()
        self.runing_refresh_similarity_list()

        list_token_ids = self.get_list_ids(position)
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, list_token_ids=list_token_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


    def set_img_pos(self, imgs_pos):
        self.imgs_pos = imgs_pos

    def _normalize_copy(self, similarity_list: torch.LongTensor) -> torch.LongTensor:
        """归一化相似度列表"""
        min_value = similarity_list.min()
        max_value = similarity_list.max()
        denom = (max_value - min_value)
        denom = denom if denom != 0 else torch.tensor(1.0, device=similarity_list.device, dtype=similarity_list.dtype)
        normalized_tensor = (similarity_list - min_value) / denom
        return normalized_tensor
    
    def _normalize(self, similarity_list: torch.LongTensor) -> torch.LongTensor:
        min_value = similarity_list.min()
        normalized_tensor = similarity_list + min_value
        return normalized_tensor
    
    def get_msg(self) -> torch.LongTensor:
        return self.msg
    
    def set_hidden_states(self, hiden_states):
        self.hidden_states = hiden_states

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token time. Used in each token position calculation."""
        time_result = 1
        for i in range(0, self.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return int(self.prf[int(time_result % self.vocab_size)])

    def set_entropy_max(self, entropy_max: float) -> None:
        self.entropy_max = entropy_max

    def set_entropy_current(self, entropy_current: float) -> None:
        self.entropy_current = entropy_current
    

class VLA(BaseWatermark):
    logitsprocessor: VLALogitsProcessor = None
    tokenizer = None
    is_random_only = False
    def __init__(self, processor = None, embedding_matrix: torch.FloatTensor = None, similarity_scheme: str = "cosine", 
                 input_embeddings: torch.LongTensor = None, model = None, tokenizer = None, 
                 transformers_config = None, split_x: int = 2, model_name: str = None, 
                 atten_delta = 0.02 , sct_radio =0.025, alpha = 0.4, trans_type = None, attn_threshold = 0.15, *args, **kwargs) -> None:
        if processor is not None:
            self.processor = processor
            special_tokens = [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id] if processor.tokenizer.pad_token_id is None else [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id, processor.tokenizer.pad_token_id]
            self.tokenizer = processor.tokenizer
            self.model = model

            self.logitsprocessor = VLALogitsProcessor(
                vocab_size=processor.tokenizer.vocab_size, 
                    embedding_matrix=embedding_matrix, 
                    similarity_scheme=similarity_scheme, 
                    special_tokens=special_tokens, 
                    split_x=split_x,
                    atten_delta=atten_delta,
                    sct_radio=sct_radio,
                    alpha=alpha,
                    trans_type=trans_type,
                    attn_threshold=attn_threshold
                )
        elif tokenizer is not None:
            self.tokenizer = tokenizer
            special_tokens = [tokenizer.eos_token_id, tokenizer.bos_token_id] if tokenizer.pad_token_id is None else [
                tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]
            self.model = model

            self.logitsprocessor = VLALogitsProcessor(
                    vocab_size=tokenizer.vocab_size, 
                    embedding_matrix=embedding_matrix,
                    similarity_scheme=similarity_scheme,
                    special_tokens=special_tokens, 
                    split_x=split_x,
                    atten_delta=atten_delta,
                    sct_radio=sct_radio,
                    alpha=alpha,
                    trans_type = trans_type,
                    attn_threshold=attn_threshold
                )

        if input_embeddings is not None:
            self.refresh_logits_processor(input_embeddings=input_embeddings)

        if transformers_config is not None:
            self.generation_model = transformers_config.model
            self.generation_tokenizer = transformers_config.tokenizer
            self.gen_kwargs = transformers_config.gen_kwargs

        self.model_name = model_name

    def refresh_logits_processor(
            self, 
            input_embeddings: torch.LongTensor, 
            hidden_states: torch.LongTensor,
            atten_scores: torch.LongTensor,
            image_pos ) -> None:
        self.logitsprocessor.set_hidden_states(hidden_states)
        self.logitsprocessor.set_img_pos(image_pos)
        self.logitsprocessor.set_attention_scores(atten_scores)
        self.logitsprocessor.refresh_msg(input_embeddings)
