import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from layers import SwinTransformer
from transformers import BertModel, CLIPModel


class System1(nn.Module):
    def __init__(self):
        super(System1, self).__init__()

        # Visual encoders
        self.st_img = SwinTransformer()
        self.st_ela = SwinTransformer()

        # Textual encoders
        self.bert =  BertModel.from_pretrained(config.bert_dir)
        self.clip = CLIPModel.from_pretrained(config.clip_dir)
        # self.gru = nn.GRU(config.text_dim, config.hidden_dim // 2, num_layers=1,
        #                   batch_first=True, bidirectional=True)

        # Matching networks
        self.contextual_match_nn = nn.Sequential(
            nn.Linear(512 * 2, config.hidden_dim),
            nn.ReLU()
        )
        self.semantic_match_nn = nn.Sequential(
            nn.Linear(768 * 2, config.hidden_dim),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Optional freezing
        self._freeze_module(self.bert, config.sys1_bert_freeze)
        self._freeze_module(self.clip, config.sys1_clip_freeze)
        self._freeze_module(self.st_img,config.sys1_swintransformer_freeze)
        self._freeze_module(self.st_ela, config.sys1_swintransformer_freeze)

    def _freeze_module(self, module, freeze: bool):
        """Freeze module parameters if freeze=True"""
        if freeze:
            for param in module.parameters():
                param.requires_grad = False

    def _get_bert_out(self, encoded_text):
        outputs = self.bert(**encoded_text)['last_hidden_state']
        return outputs[:, 0, :]  #[CLS]

    def forward(self, data):
        qCap, qImg, ELA_img, img_to_text, clip_input = data

        # CLIP contextual matching
        clip_outputs = self.clip(**clip_input)
        img_embed = clip_outputs.image_embeds  # [B, 512]
        text_embed = clip_outputs.text_embeds  # [B, 512]
        contextual_match = self.contextual_match_nn(torch.cat([img_embed, text_embed], dim=-1))

        # Semantic matching
        qcap_feat = self._get_bert_out(qCap)         # [B, 768]
        img2text_feat = self._get_bert_out(img_to_text)  # [B, 768]
        semantic_match = self.semantic_match_nn(torch.cat([qcap_feat, img2text_feat], dim=-1))

        # Visual features
        qImg_feat = self.st_img(qImg).mean(dim=1)       # [B, 768]
        ELA_img_feat = self.st_ela(ELA_img).mean(dim=1) # [B, 768]

        # Final classification
        fused = torch.cat([qcap_feat, qImg_feat, ELA_img_feat, contextual_match, semantic_match], dim=-1)
        logits = self.classifier(fused)

        return logits
