import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from layers import SwinTransformer
from transformers import BertModel


class Scheduler(nn.Module):
    def __init__(self):
        super(Scheduler, self).__init__()
        # Initialize Swin Transformer for image feature extraction
        self.st_img = SwinTransformer()
        # Initialize pretrained BERT for text encoding
        self.bert = BertModel.from_pretrained(config.bert_dir)

        # Classifier head to output logits for 2 classes (System1 or System2 dispatch)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, 2)
        )

        # Optionally freeze BERT and SwinTransformer weights based on config flags
        self._freeze_module(self.bert, config.scheduler_bert_freeze)
        self._freeze_module(self.st_img, config.scheduler_swintransformer_freeze)

    def _freeze_module(self, module, freeze: bool):
        """Freeze module parameters if freeze=True."""
        if freeze:
            for param in module.parameters():
                param.requires_grad = False

    def _get_bert_out(self, encoded_text):
        """
        Extract CLS token representation from BERT output.

        Args:
            encoded_text (dict): Tokenized input dictionary for BERT

        Returns:
            Tensor of shape [batch_size, 768] representing CLS token embeddings
        """
        outputs = self.bert(**encoded_text)['last_hidden_state']
        return outputs[:, 0, :]  # CLS token embedding

    def forward(self, data):
        """
        Forward pass for Scheduler.

        Args:
            data (tuple): (qCap, qImg)
                qCap: tokenized text input for BERT
                qImg: image tensor input for SwinTransformer

        Returns:
            logits tensor of shape [batch_size, 2], indicating dispatch decision
        """
        qCap, qImg = data
        qcap_feat = self._get_bert_out(qCap)  # Text features from BERT [B, 768]
        qImg_feat = self.st_img(qImg).mean(dim=1)  # Image features from SwinTransformer [B, 768]
        fused = torch.cat([qcap_feat, qImg_feat], dim=-1)  # Concatenate text and image features
        logits = self.classifier(fused)  # Predict dispatch logits

        return logits
