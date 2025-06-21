import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
import config
from transformers import BertModel, CLIPModel
from layers import SwinTransformer, SGATConv, CrossAttention, CoAttention


class System2(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, num_heads=2, n_layers=2, residual=False):
        super(System2, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.text_dim = config.text_dim
        self.img_dim = config.img_dim

        # Multimodal encoders
        self.bert =  BertModel.from_pretrained(config.bert_dir)
        self.clip = CLIPModel.from_pretrained(config.clip_dir)
        self.st_img = SwinTransformer()
        self.st_ela = SwinTransformer()

        # Freeze pretrained models if configured
        self._freeze_module(self.bert, config.sys2_bert_freeze)
        self._freeze_module(self.clip, config.sys2_clip_freeze)
        self._freeze_module(self.st_img,config.sys2_swintransformer_freeze)
        self._freeze_module(self.st_ela, config.sys2_swintransformer_freeze)

        # GRU for textual processing
       # self.gru = nn.GRU(self.text_dim, self.text_dim // 2, batch_first=True, bidirectional=True)

        # Semantic & Contextual Matching
        self.contextual_match_nn = nn.Sequential(
            nn.Linear(512 * 2, self.hidden_dim),
            nn.ReLU()
        )
        self.semantic_match_nn = nn.Sequential(
            nn.Linear(768 * 2, self.hidden_dim),
            nn.ReLU()
        )

        # SGAT Layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-attention for evidence
        self.cross_attention_text = CrossAttention(self.text_dim, self.text_dim, self.text_dim,
                                                   num_heads=config.att_num_heads,
                                                   dropout=config.att_dropout)
        self.cross_attention_image = CrossAttention(self.img_dim, self.img_dim, self.img_dim,
                                                    num_heads=config.att_num_heads,
                                                    dropout=config.att_dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _freeze_module(self, module, freeze=True):
        if freeze:
            for param in module.parameters():
                param.requires_grad = False

    def _get_bert_out(self, bert_inputs):
        bert_output = self.bert(**bert_inputs)['last_hidden_state']
        return bert_output[:, 0, :], bert_output

    def _mean_swin_feature(self, img_batch):
        """Apply SwinTransformer and average pooling over patch dimension"""
        return self.st_img(img_batch).mean(dim=1)

    def forward(self, data):
        qCap, qImg, ELA_img, img_to_text, clip_input, t_evidence, v_evidence, text_dgl, img_dgl = data

        text_dgl=dgl.batch(text_dgl)
        img_dgl = dgl.batch(img_dgl)

        # 1. CLIP Matching
        clip_outputs = self.clip(**clip_input)
        contextual_match = self.contextual_match_nn(
            torch.cat([clip_outputs.image_embeds, clip_outputs.text_embeds], dim=-1))

        # 2. Semantic Matching
        qcap_feat, qcap_seq = self._get_bert_out(qCap)
        img2text_feat, _ = self._get_bert_out(img_to_text)
        semantic_match = self.semantic_match_nn(torch.cat([qcap_feat, img2text_feat], dim=-1))

        # 3. Visual Features
        qImg_seq = self.st_img(qImg)  # [B, 49, 768]
        qImg_feat = qImg_seq.mean(dim=1)
        ela_feat = self.st_ela(ELA_img).mean(dim=1)

        # 4. Textual Evidence Alignment
        t_evi_seq = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence], dim=0)
        textual_evidence = self.cross_attention_text(qcap_seq, t_evi_seq).mean(dim=1)

        # 5. Visual Evidence Alignment
        v_evi_seq = torch.stack([self._mean_swin_feature(img) for img in v_evidence], dim=0)
        visual_evidence = self.cross_attention_image(qImg_seq, v_evi_seq).mean(dim=1)

        # 6. SGAT DGL Graph Features
        img_feats = img_dgl.ndata['x']
        text_feats = text_dgl.ndata['x']

        # 保证边特征存在
        if 'x' not in img_dgl.edata:
            img_dgl.edata['x'] = torch.zeros((img_dgl.number_of_edges(), 768),
                                             device=img_feats.device)
        if 'x' not in text_dgl.edata:
            text_dgl.edata['x'] = torch.zeros((text_dgl.number_of_edges(), 768),
                                              device=text_feats.device)

        for layer in self.SGAT_layers:
            img_feats = layer(img_dgl, img_feats, img_dgl.edata['x']).mean(dim=1)
            text_feats = layer(text_dgl, text_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_feats
        text_dgl.ndata['h'] = text_feats

        img_dgl_feat = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feat = dgl.readout_nodes(text_dgl, 'h', op='mean')
        sgat_feat = F.leaky_relu(self.dgl_linear(torch.cat([img_dgl_feat, text_dgl_feat], dim=-1)))

        # (可选) 协同注意力可加入此处
        # co_it, co_ti = self.co_attention(img_dgl_feat, text_dgl_feat)
        # co_fea = torch.stack([co_it.mean(dim=1), co_ti.mean(dim=1)], dim=1).mean(dim=1)

        # 7. Classification
        fusion = torch.cat([
            qcap_feat, qImg_feat, ela_feat,
            contextual_match, semantic_match,
            textual_evidence, visual_evidence,
            sgat_feat
        ], dim=-1)

        return self.classifier(fusion)