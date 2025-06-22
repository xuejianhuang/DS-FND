import torch
import config
import dgl


def collate_sys1(batch):
    samples, keys = zip(*batch)

    labels = torch.stack([s['label'] for s in samples]).to(config.device)
    qImg_batch = torch.stack([s['qImg'] for s in samples]).to(config.device)
    img_batch = [s['img'] for s in samples]
    qCap_texts = [s['qCap'] for s in samples]
    ELA_img_batch = torch.stack([s['ELA_img'] for s in samples]).to(config.device)
    img_to_text_texts = [s['img_to_text'] for s in samples]

    clip_input_batch = config._clip_processor(
        text=qCap_texts, images=img_batch,
        return_tensors="pt", padding=True, truncation=True
    ).to(config.device)

    qCap_batch = config._tokenizer(
        qCap_texts, return_tensors='pt', max_length=config.text_max_length,
        padding='max_length', truncation=True
    ).to(config.device)

    img_to_text_batch = config._tokenizer(
        img_to_text_texts, return_tensors='pt', max_length=config.text_max_length,
        padding='max_length', truncation=True
    ).to(config.device)

    return labels, qCap_batch, qImg_batch, ELA_img_batch, img_to_text_batch, clip_input_batch, keys


def collate_sys2(batch):
    samples = [item[0] for item in batch]
    max_t_len = max(item[1] for item in batch)
    max_i_len = max(item[2] for item in batch)
    keys = [item[3] for item in batch]

    labels = []
    qImg_batch, img_batch, qCap_batch, ELA_img_batch = [], [], [], []
    img_to_text_batch, i_evidence_batch, t_evidence_batch = [], [], []
    img_dgl_graphs, text_dgl_graphs = [], []

    for sample in samples:
        labels.append(sample['label'])

        # Process textual evidence
        t_evidence = sample['t_evidence'][:max_t_len] + [""] * (max_t_len - len(sample['t_evidence']))
        t_encoded = config._tokenizer(
            t_evidence, return_tensors='pt', max_length=config.text_max_length,
            padding='max_length', truncation=True
        ).to(config.device)
        t_evidence_batch.append(t_encoded)

        # Process image evidence with padding if necessary
        i_evidence = sample['i_evidence']
        if len(sample['i_evidence'].shape) > 2:
            pad_size = (max_i_len - i_evidence.shape[0], *i_evidence.shape[1:])
        else:
            pad_size = (max_i_len, 3, 224, 224)
        padded_imgs = torch.cat([i_evidence, torch.zeros(pad_size, device=i_evidence.device)], dim=0)
        i_evidence_batch.append(padded_imgs)

        # Collect other fields
        qImg_batch.append(sample['qImg'])
        img_batch.append(sample['img'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])
        img_to_text_batch.append(sample['img_to_text'])
        img_dgl_graphs.append(sample['img_dgl_graph'].to(config.device))
        text_dgl_graphs.append(sample['text_dgl_graph'].to(config.device))

    clip_input_batch = config._clip_processor(
        text=qCap_batch, images=img_batch,
        return_tensors="pt", padding=True, truncation=True
    ).to(config.device)

    qCap_batch = config._tokenizer(
        qCap_batch, return_tensors='pt', max_length=config.text_max_length,
        padding='max_length', truncation=True
    ).to(config.device)

    img_to_text_batch = config._tokenizer(
        img_to_text_batch, return_tensors='pt', max_length=config.text_max_length,
        padding='max_length', truncation=True
    ).to(config.device)

    return (
        torch.stack(labels).to(config.device),
        qCap_batch,
        torch.stack(qImg_batch).to(config.device),
        torch.stack(ELA_img_batch).to(config.device),
        img_to_text_batch,
        clip_input_batch,
        t_evidence_batch,
        torch.stack(i_evidence_batch).to(config.device),
        text_dgl_graphs,
        img_dgl_graphs,
        # dgl.batch(text_dgl_graphs).to(config.device),
        # dgl.batch(img_dgl_graphs).to(config.device),
        keys
    )

def collate_scheduler(batch):
    samples, keys = zip(*batch)
    labels = torch.stack([s['label'] for s in samples]).to(config.device)
    qImg_batch = torch.stack([s['qImg'] for s in samples]).to(config.device)
    qCap_texts = [s['qCap'] for s in samples]

    qCap_batch = config._tokenizer(
        qCap_texts, return_tensors='pt', max_length=config.text_max_length,
        padding='max_length', truncation=True
    ).to(config.device)

    return labels, qCap_batch, qImg_batch, keys

