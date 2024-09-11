import torch
import torch.nn as nn
from transformers import BertModel


# tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-PT_laptop")

class Net(nn.Module):
    def __init__(self, device='cpu',num_labels=3):
        super().__init__()
        self.num_labels=num_labels
        # self.bert = AutoModel.from_pretrained("activebus/BERT-PT_laptop")
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)

        self.device = device
        # self.finetuning = finetuning

    def forward(self, x, inp_msk,seg_id,y=None ):
        labels=y
        # if y!=None:

        x = x.to(self.device)
        labels = y.to(self.device)
        seg_id = seg_id.to(self.device)
        inp_msk= inp_msk.to(self.device)

        if self.training:
            self.bert.train()
            output = self.bert(x,inp_msk,seg_id)
            enc = output[0]
            enc = self.dropout(enc)
            # print("->bert.train()1")
        else:
            self.bert.eval()
            with torch.no_grad():
                output= self.bert(x,inp_msk,seg_id)
                enc = output[0]
                # print("->bert.train()2")
        logits = self.fc(enc)
        # print("->bert.train()3")
        y_hat = logits.argmax(-1)
        # print("->bert.train()4")
        # return logits, y, y_hat
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return enc,logits,y,y_hat,loss
        else:
            return logits


