import argparse
import torch

from dataloader import DataLoader
from hyperparameters import Hyperparameters
from model import GPTLanguageModel
from tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run inference with the GPT model."
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval"],
        help="Mode to run the script in: train, eval.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default="model.pth",
        type=str,
        help="Path to the output file to write the model checkpoint to.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output file to write the generated text to.",
    )
    parser.add_argument(
        "--inferece-token-size",
        type=int,
        default=2000,
        help="Number of tokens to generate during inference.",
    )
    args = parser.parse_args()

    hyperparameters = Hyperparameters()
    dataloader = DataLoader(data_path="./dataset/data.txt")
    tokenizer = SimpleTokenizer(dataloader.get_unique_chars())
    model = GPTLanguageModel(
        config=hyperparameters, token_size=dataloader.get_vocab_size()
    )
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        model = torch.load(args.checkpoint, weights_only=False)

    m = model.to(hyperparameters.device)
    print("Using device: ", hyperparameters.device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    if args.mode == "train":
        train(
            model=model,
            hyperparameters=hyperparameters,
            dataloader=dataloader,
            tokenizer=tokenizer,
            output=args.checkpoint_output,
        )
    elif args.mode == "eval":
        evaluate(
            model=model,
            tokenizer=tokenizer,
            hyperparameters=hyperparameters,
            token_size=args.inferece_token_size,
            output=args.output,
        )


def train(model, hyperparameters, dataloader, tokenizer, output):
    print("Training the model...")
    data = torch.tensor(tokenizer.encode(dataloader.get_data()), dtype=torch.long)
    nm = int(len(data) * 0.9)
    train_data = data[:nm]
    val_data = data[nm:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate)
    for steps in range(hyperparameters.max_iter):
        if (
            steps % hyperparameters.eval_every == 0
            or steps == hyperparameters.max_iter - 1
        ):
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                hyperparameters=hyperparameters,
            )
            print(
                f"Step: {steps}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}"
            )
        xb, yb = get_batch(
            train_data,
            hyperparameters.block_size,
            hyperparameters.batch_size,
            hyperparameters.device,
        )
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model, output)
    evaluate(model, hyperparameters, tokenizer, 1000)


def evaluate(model, hyperparameters, tokenizer, token_size=2000, output=None):
    print("Evaluating the model...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=hyperparameters.device)
    if output:
        with open(output, "w") as f:
            f.write(
                tokenizer.decode(
                    model.generate(context, max_new_tokens=token_size)[0].tolist()
                )
            )
    else:
        print(
            tokenizer.decode(
                model.generate(context, max_new_tokens=token_size)[0].tolist()
            )
        )


def get_batch(data, block_size, batch_size, device):
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, hyperparameters):
    out = {}
    model.eval()
    for key, data in {"train": train_data, "val": val_data}.items():
        lossess = torch.zeros(hyperparameters.eval_iter)
        for i in range(hyperparameters.eval_iter):
            X, Y = get_batch(
                data,
                hyperparameters.block_size,
                hyperparameters.batch_size,
                hyperparameters.device,
            )
            y_pred, loss = model(X, Y)
            lossess[i] = loss.item()
        out[key] = lossess.mean()
    model.train()
    return out


if __name__ == "__main__":
    main()
