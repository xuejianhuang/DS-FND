import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from dgl import load_graphs

from collate_fn import collate_sys1, collate_sys2, collate_scheduler
from utils.util import (
    get_transform_Compose, get_transform_Compose_ELA, get_ela_image_path,
    load_img_pil, load_captions, load_captions_weibo, load_imgs_direct_search,
    split_dataset, load_dispatch_labels
)
from model.system1 import System1
from model.system2 import System2
from model.scheduler import Scheduler
import config

class BaseDataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.data_items = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.data_items.keys())
        self.img_transform = get_transform_Compose()
        self.ela_transform = get_transform_Compose_ELA()

    def __len__(self):
        return len(self.data_items)

    def get_item_common(self, key):
        item = self.data_items[key]
        label = torch.tensor(int(item['label']))
        caption = item['caption']
        img_to_text = item['img-to-text']
        img_path = os.path.join(self.data_root_dir, item['image_path'])
        ela_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        pil_img = load_img_pil(img_path)
        ela_img = self.ela_transform(load_img_pil(ela_path))
        return label, caption, img_to_text, pil_img, self.img_transform(pil_img), ela_img

class System1Dataset(BaseDataset):
    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        label, caption, img_to_text, pil_img, q_img, ela_img = self.get_item_common(key)
        return {
            'label': label,
            'qImg': q_img,
            'img': pil_img,
            'qCap': caption,
            'ELA_img': ela_img,
            'img_to_text': img_to_text
        }, key

class System2Dataset(BaseDataset):
    def __init__(self, data_items, data_root_dir, knowledge_enhanced=True):
        super().__init__(data_items, data_root_dir)
        graph_dir = os.path.join(data_root_dir, 'graph')
        self.img_graphs = load_graphs(os.path.join(graph_dir,
            'knowledge-enhanced_img_dgl_graph.bin' if knowledge_enhanced else 'img_dgl_graph.bin'))[0]
        self.text_graphs = load_graphs(os.path.join(graph_dir,
            'knowledge-enhanced_text_dgl_graph.bin' if knowledge_enhanced else 'text_dgl_graph.bin'))[0]

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.data_items[key]
        label, caption, img_to_text, pil_img, q_img, ela_img = self.get_item_common(key)

        # Load visual and textual evidences
        try:
            direct_path = os.path.join(self.data_root_dir, item['direct_path'])
            inv_path = os.path.join(self.data_root_dir, item['inv_path'])

            with open(os.path.join(inv_path, 'inverse_annotation.json'), encoding='utf-8') as f:
                inv_ann = json.load(f)
            with open(os.path.join(direct_path, 'direct_annotation.json'), encoding='utf-8') as f:
                direct_ann = json.load(f)

            t_evidence = load_captions(inv_ann) + load_captions_weibo(direct_ann)
            t_evidence = t_evidence[:config.max_captions_num]
            i_evidence = load_imgs_direct_search(self.img_transform, direct_path, direct_ann, config.max_images_num)
        except Exception as e:
            print(f"[Error] Loading evidence for key {key}: {e}")
            t_evidence, i_evidence = [], torch.zeros((0, 3, 224, 224))

        return {
            'label': label,
            'qImg': q_img,
            'img': pil_img,
            'qCap': caption,
            'ELA_img': ela_img,
            'img_to_text': img_to_text,
            't_evidence': t_evidence,
            'i_evidence': i_evidence,
            'img_dgl_graph': self.img_graphs[int(key)],
            'text_dgl_graph': self.text_graphs[int(key)],
        }, len(t_evidence), i_evidence.shape[0], key

class SchedulerDataset(Dataset):
    def __init__(self, data_items, data_root_dir,labels):
        self.data_items = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.data_items.keys())
        self.img_transform = get_transform_Compose()
        self.labels=labels

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.data_items.get(key)
        caption = item['caption']
        img_path = os.path.join(self.data_root_dir, item['image_path'])
        pil_img = load_img_pil(img_path)
        q_img=self.img_transform(pil_img)
        label = torch.tensor(self.labels[key])
        return {
                   'label': label,
                   'qImg': q_img,
                   'qCap': caption
               }, key

    def __len__(self):
        return len(self.data_items)
    

def get_model_and_data(model_name='System1', dataset='weibo', batch=config.batch_size):
    data_root_dir = config.weibo_dataset_dir if dataset == 'weibo' else config.twitter_dataset_dir
    dataset_path = os.path.join(data_root_dir, 'dataset_items_merged.json')
    train_items, val_items, test_items = split_dataset(dataset_path, config.train_ratio, config.val_ratio)

    if model_name == 'System1':
        train_ds = System1Dataset(train_items, data_root_dir)
        val_ds = System1Dataset(val_items, data_root_dir)
        test_ds = System1Dataset(test_items, data_root_dir)
        model = System1().to(config.device)
        collate_fn = collate_sys1

    elif model_name == 'System2':
        train_ds = System2Dataset(train_items, data_root_dir)
        val_ds = System2Dataset(val_items, data_root_dir)
        test_ds = System2Dataset(test_items, data_root_dir)
        model = System2(config.node_feats, config.edge_feats, config.out_feats,
                        config.num_heads, config.n_layers).to(config.device)
        collate_fn = collate_sys2
    elif model_name == "Scheduler":
        csv_path = f"./outputs/{dataset}_dispatch_labels.csv"
        labels = load_dispatch_labels(csv_path)
        train_ds = SchedulerDataset(train_items, data_root_dir,labels=labels)
        val_ds =SchedulerDataset(val_items, data_root_dir,labels=labels)
        test_ds = SchedulerDataset(test_items, data_root_dir,labels=labels)
        model = Scheduler().to(config.device)
        collate_fn = collate_scheduler
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)

    return model, train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset_path = os.path.join(config.twitter_dataset_dir, 'dataset_items_merged.json')
    train, val, test = split_dataset(dataset_path, config.train_ratio, config.val_ratio)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")