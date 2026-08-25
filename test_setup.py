from pathlib import Path

from src.model.config import ModelConfig


ROOT = Path(__file__).resolve().parent

config = ModelConfig()

print("Sarah project initialized")
print()
print("Project:", ROOT)
print("Vocabulary:", config.vocab_size)
print("Context:", config.max_seq_len)
print("Hidden size:", config.hidden_size)
print("Layers:", config.num_layers)
print("Attention heads:", config.num_heads)
print("FFN size:", config.intermediate_size)
