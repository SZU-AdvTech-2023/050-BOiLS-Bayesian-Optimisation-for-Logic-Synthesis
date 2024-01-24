from typing import Callable, Optional

import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch


class InputTransformation(ABC):

    def __init__(self, id: str, embed_dim: int):
        self.id = id
        self.embed_dim = embed_dim

    @abstractmethod
    def __call__(self, x):
        assert False, "Should implement"


class SentenceBertInputTransform(InputTransformation):

    def __init__(self, sbert_model, id: str, name_embed: bool, name_embedder: Optional[Callable[[int], str]]):
        self.sbert_model: SentenceTransformer = sbert_model
        self.name_embed = name_embed
        self.name_embedder = name_embedder
        embed_dim = self.sbert_model.get_sentence_embedding_dimension()
        super().__init__(id, embed_dim=embed_dim)

    def __call__(self, input: np.ndarray) -> Tensor:
        assert input.ndim == 2, f"Input should be of ndim 2, got shape: {input.shape}"
        if self.name_embed:
            assert self.name_embedder is not None
            input_str = np.array([", ".join(map(self.name_embedder, list(xi))) for xi in input])
        else:
            input_str = np.array([", ".join(map(str, list(xi))) for xi in input])
        return torch.from_numpy(self.sbert_model.encode(input_str))
