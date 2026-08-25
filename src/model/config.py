from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048

    hidden_size: int = 896
    num_layers: int = 19
    num_heads: int = 14

    intermediate_size: int = 2560

    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
