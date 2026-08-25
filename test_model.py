import torch

from src.model.config import ModelConfig
from src.model.transformer import SarahModel


def main():
    config = ModelConfig()

    print("Creating Sarah...")
    model = SarahModel(config)

    params = model.count_parameters()

    print()
    print("Parameters:", f"{params:,}")
    print("Parameters (M):", round(params / 1_000_000, 2))

    batch_size = 1
    seq_len = 64

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len)
    )

    print()
    print("Input:", input_ids.shape)

    logits = model(input_ids)

    print("Logits:", logits.shape)

    targets = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len)
    )

    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1)
    )

    print("Loss:", loss.item())

    loss.backward()

    print("Backward: OK")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
