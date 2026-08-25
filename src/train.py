import torch
from torch.utils.data import Dataset, DataLoader

from src.model.config import ModelConfig
from src.model.transformer import SarahModel


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SEQ_LEN = 256
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

TEST_STEPS = 20
SAVE_EVERY = 10
LOG_EVERY = 1


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def set_seed():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def make_test_tokens(vocab_size, count):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return torch.randint(
        0,
        vocab_size,
        (count,),
        generator=generator
    ).tolist()


def save_checkpoint(model, optimizer, step, loss):
    path = CHECKPOINT_DIR / f"test_step_{step}.pt"

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        },
        path
    )

    print("Checkpoint:", path)


def train():
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2
            ),
            "GB"
        )

    config = ModelConfig()

    model = SarahModel(config).to(device)

    parameters = model.count_parameters()

    print("Parameters:", parameters)
    print(
        "Parameters (M):",
        round(parameters / 1_000_000, 2)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    tokens = make_test_tokens(
        config.vocab_size,
        SEQ_LEN * 30
    )

    dataset = TokenDataset(
        tokens,
        SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    model.train()

    optimizer.zero_grad(set_to_none=True)

    step = 0

    print("Starting test training...")

    while step < TEST_STEPS:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):
                logits = model(x)

                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    y.reshape(-1)
                )

                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaled_loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1

            if step % LOG_EVERY == 0:
                print(
                    f"step={step:03d} "
                    f"loss={loss.item():.6f}"
                )

            if step % SAVE_EVERY == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    loss.item()
                )

            if step >= TEST_STEPS:
                break

    print("Test training finished.")


if __name__ == "__main__":
    train()
