import time, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from torch.cuda.amp import autocast, GradScaler

from src.data_utils import load_sst2, load_qqp, load_stsb
from src.model_utils import MultiTaskSLM
from src.utils import set_seed, log_to_csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 2
BATCH_SIZE = 64
LR = 2e-5

def run(task):
    set_seed()

    # ---- LOAD DATA (SLM tokenizer) ----
    if task == "sst2":
        data = load_sst2("slm")
    elif task == "qqp":
        data = load_qqp("slm")
    else:
        data = load_stsb("slm")

    train_loader = DataLoader(
        data["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        data["validation"],
        batch_size=BATCH_SIZE
    )

    model = MultiTaskSLM({"sst2":2,"qqp":2,"stsb":1}).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"\n===== START SINGLE-TASK TRAINING: {task.upper()} =====")

    start_total = time.time()

    # ---------- TRAIN ----------
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

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
    print(f"TOTAL TRAIN TIME: {train_time/60:.2f} minutes")

    # ---------- EVALUATION ----------
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            out = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                task
            )
            preds.append(out.cpu())
            labels.append(batch["label"])

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    row = {
        "model_type":"single-slm",
        "model_name":"distilbert",
        "task":task,
        "accuracy":"",
        "f1":"",
        "pearson":"",
        "spearman":"",
        "train_time":train_time,
        "params":sum(p.numel() for p in model.parameters())
    }

    if task == "stsb":
        row["pearson"] = pearsonr(preds.squeeze(), labels)[0]
        row["spearman"] = spearmanr(preds.squeeze(), labels)[0]
    else:
        row["accuracy"] = accuracy_score(labels, preds.argmax(1))
        row["f1"] = f1_score(labels, preds.argmax(1), average="macro")

    log_to_csv(f"logs/single_{task}.csv", row)

    print(f"Results saved to logs/single_{task}.csv")

# ------------------------------------------------

if __name__ == "__main__":
    for task in ["sst2","qqp","stsb"]:
        run(task)
