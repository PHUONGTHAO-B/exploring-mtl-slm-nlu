import time, torch, random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from torch.cuda.amp import autocast, GradScaler

from src.data_utils import load_sst2, load_qqp, load_stsb
from src.model_utils import MultiTaskLLM
from src.utils import set_seed, log_to_csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 2
BATCH_SIZE = 64
LR = 2e-5

def main():
    set_seed()

    datasets = {
        "sst2": load_sst2("llm"),
        "qqp": load_qqp("llm"),
        "stsb": load_stsb("llm")
    }

    train_loaders = {
        t: DataLoader(
            d["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        for t,d in datasets.items()
    }

    val_sets = {t:d["validation"] for t,d in datasets.items()}

    model = MultiTaskLLM({"sst2":2,"qqp":2,"stsb":1}).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    print("===== START MULTI-TASK LLM TRAINING =====")

    start_total = time.time()
    iters = {t:cycle(l) for t,l in train_loaders.items()}

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for _ in tqdm(
            range(max(len(l) for l in train_loaders.values())),
            desc=f"Epoch {epoch+1}"
        ):
            task = random.choice(list(iters.keys()))
            batch = next(iters[task])

            optimizer.zero_grad()

            with autocast():
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    task
                )

                if task == "stsb":
                    loss = torch.nn.functional.mse_loss(
                        logits.squeeze(),
                        batch["label"].to(DEVICE).float()
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        batch["label"].to(DEVICE)
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch+1} finished in {(time.time()-epoch_start)/60:.2f} minutes")

    train_time = time.time() - start_total
    print(f"\nTOTAL TRAIN TIME: {train_time/60:.2f} minutes")

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    for task, ds in val_sets.items():
        loader = DataLoader(ds, batch_size=BATCH_SIZE)
        preds, labels = [], []

        with torch.no_grad():
            for b in loader:
                out = model(
                    b["input_ids"].to(DEVICE),
                    b["attention_mask"].to(DEVICE),
                    task
                )
                preds.append(out.cpu())
                labels.append(b["label"])

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        row = {
            "model_type":"multi-llm",
            "model_name":"bert-base",
            "task":task,
            "accuracy":"",
            "f1":"",
            "pearson":"",
            "spearman":"",
            "train_time":train_time,
            "params":num_params
        }

        if task == "stsb":
            row["pearson"] = pearsonr(preds.squeeze(), labels)[0]
            row["spearman"] = spearmanr(preds.squeeze(), labels)[0]
        else:
            row["accuracy"] = accuracy_score(labels, preds.argmax(1))
            row["f1"] = f1_score(labels, preds.argmax(1), average="macro")

        log_to_csv("logs/multi_llm.csv", row)

    print("Results saved to logs/multi_llm.csv")

if __name__ == "__main__":
    main()
