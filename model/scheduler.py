import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from layers import SwinTransformer
from transformers import BertModel


class Scheduler(nn.Module):
    def __init__(self):
        super(Scheduler, self).__init__()
        self.st_img = SwinTransformer()
        self.bert = BertModel.from_pretrained(config.bert_dir)


        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, 2)
        )

        # Optional freezing
        self._freeze_module(self.bert, config.scheduler_bert_freeze)
        self._freeze_module(self.st_img,config.scheduler_swintransformer_freeze)


    def _freeze_module(self, module, freeze: bool):
        """Freeze module parameters if freeze=True"""
        if freeze:
            for param in module.parameters():
                param.requires_grad = False

    def _get_bert_out(self, encoded_text):
        outputs = self.bert(**encoded_text)['last_hidden_state']
        return outputs[:, 0, :]  #[CLS]

    def forward(self, data):
        qCap, qImg = data
        qcap_feat = self._get_bert_out(qCap)         # [B, 768]
        qImg_feat = self.st_img(qImg).mean(dim=1)       # [B, 768]
        fused = torch.cat([qcap_feat, qImg_feat], dim=-1)
        logits = self.classifier(fused)

        return logits
