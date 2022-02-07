from transformers import AutoModel
import torch.nn as nn

class SentiBERT(nn.Module):

    def __init__(self, **kargs):
        super(SentiBERT, self).__init__()

        hidden_dim = kargs['hidden_dim']
        self.bert = AutoModel.from_pretrained(kargs['model_name_or_path'])

        for param in self.bert.parameters():
            param.requires_grad = False

        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU())
        self.layer2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        """
        x: x is a list of token tensors, already tokenized by the tokenizer
        """
        x = self.bert(x)
        last_hidden_state_cls = x.last_hidden_state[:, 0, :]

        x = self.layer1(torch.squeeze(last_hidden_state_cls))
        out = self.layer2(x)

        return out