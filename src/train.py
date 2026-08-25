import math
import time
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.model.config import SarahConfig
from src.model.transformer import Sarah


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
SEQ_LEN = 2048

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MAX_STEPS = 1000
SAVE_EVERY = 250
LOG_EVERY = 10

VAL_FRACTION = 0.02


class TokenDataset(Dataset):
    """
    Временный dataset-интерфейс.

    Позже сюда подключим настоящий tokenizer/dataset.
    Сейчас класс специально ожидает уже готовые token IDs.
    """

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(
            0,
            (len(self.tokens) - 1) // self.seq_len
        )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        if len(chunk) < self.seq_len + 1:
            raise IndexError

        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long
        )

        return x, y


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(device):
    config = SarahConfig()

    model = Sarah(config)

    model.to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
    print(
        f"Parameters (M): "
        f"{parameters / 1_000_000:.2f}"
    )

    return model


def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def save_checkpoint(
    model,
    optimizer,
    scaler,
    step,
    loss,
):
    path = CHECKPOINT_DIR / f"step_{step}.pt"

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    scaler.load_state_dict(
        checkpoint["scaler"]
    )

    step = checkpoint["step"]
    loss = checkpoint["loss"]

    print(
        f"Resumed from step {step}"
    )

    return step, loss


def fake_tokens(vocab_size, count):
    """
    Временные случайные токены.

    Нужны только для проверки training loop.
    Это НЕ обучающий датасет.
    """

    generator = torch.Generator()

    generator.manual_seed(SEED)

    return torch.randint(
        0,
        vocab_size,
        (count,),
        generator=generator,
    ).tolist()


def train():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = create_model(device)

    optimizer = create_optimizer(model)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    config = SarahConfig()

    # Временные данные исключительно
    # для проверки training loop.
    tokens = fake_tokens(
        config.vocab_size,
        SEQ_LEN * 20,
    )

    dataset = TokenDataset(
        tokens,
        SEQ_LEN,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    step = 0

    print("Starting training...")

    while step < MAX_STEPS:

        accumulated_loss = 0.0

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, loss = model(
                    x,
                    targets=y,
                )

                loss = (
                    loss /
                    GRAD_ACCUM_STEPS
                )

            scaler.scale(loss).backward()

            accumulated_loss += (
                loss.item() *
                GRAD_ACCUM_STEPS
            )

            if (
                (step + 1)
                % GRAD_ACCUM_STEPS
                == 0
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            step += 1

            if step % LOG_EVERY == 0:

                print(
                    f"step={step:05d} "
                    f"loss={accumulated_loss:.4f}"
                )

            if step % SAVE_EVERY == 0:

                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    accumulated_loss,
                )

            if step >= MAX_STEPS:
                break

    save_checkpoint(
        model,
        optimizer,
        scaler,
        step,
        accumulated_loss,
    )

    print("Training finished.")


if __name__ == "__main__":
    train()
