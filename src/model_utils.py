import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel

# -------- SLM ----------
class MultiTaskSLM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        h = self.encoder.config.hidden_size

        self.heads = nn.ModuleDict({
            "sst2": nn.Linear(h, 2),
            "qqp": nn.Linear(h, 2),
            "stsb": nn.Linear(h, 1)
        })

    def forward(self, input_ids, attention_mask, task):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:,0]
        return self.heads[task](cls)

# -------- LLM ----------
class MultiTaskLLM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = BertModel.from_pretrained(
            "bert-base-uncased"
        )
        h = self.encoder.config.hidden_size

        self.heads = nn.ModuleDict({
            "sst2": nn.Linear(h, 2),
            "qqp": nn.Linear(h, 2),
            "stsb": nn.Linear(h, 1)
        })

    def forward(self, input_ids, attention_mask, task):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:,0]
        return self.heads[task](cls)
