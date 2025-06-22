# 📰 Code for the Paper  
**🧠 Thinking Fast and Slow: A Dual-System Framework for Multimodal Fake News Detection via Dynamic Scheduling**

---

## 📂 Datasets

- The original dataset can be downloaded from:  
  - [Google Drive](https://drive.google.com/file/d/14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4/view?usp=sharing)  
  - [Baidu Cloud (提取码: 1odv)](https://pan.baidu.com/s/1OV_Oab0zQgI8P2Wo1qwBuw?pwd=1odv)

- Preprocessed **ELA images and scene graphs** are available via:  
  - [Baidu Cloud (提取码: 5srs)](https://pan.baidu.com/s/1AokbHvhAhqZy9-3wglzi2Q?pwd=5srs)

- For the **evidence collection process**, please refer to this repository:  
  - [https://github.com/S-Abdelnabi/OoC-multi-modal-fc](https://github.com/S-Abdelnabi/OoC-multi-modal-fc)

---

## 🚀 Quick Start

Train and evaluate the proposed models using the following commands:

### 🔧 Training

```bash
# Train System1 on Weibo dataset
python main.py --dataset weibo --model System1 --mode train

# Train System2 on Weibo dataset
python main.py --dataset weibo --model System2 --mode train
```
### 🧪 Testing

```bash
# Test System1
python main.py --dataset weibo --model System1 --mode test

# Test System2
python main.py --dataset weibo --model System2 --mode test
```

### ⚙️ Confidence-Based Scheduler
```bash
python scheduler1_infer.py --dataset weibo
```
### ⚙️ Complexity-Based Scheduler
```bash
python scheduler2_infer.py --dataset weibo
```
### 💾 Checkpoints
Pretrained model checkpoints can be downloaded from:

👉 [Baidu Cloud](https://pan.baidu.com/s/1oaHugXP7LAhFbXrPjvOeCg?pwd=4n3q) (pwd: 4n3q) 

## 📦 Dependencies
Below is a list of required packages:
```txt
tqdm==4.63.1
torch==1.12.1+cu113
seaborn==0.11.2
matplotlib==3.3.4
Pillow==8.4.0
pytorch_lightning==1.9.0
transformers==4.18.0
pycocotools
torchmetrics==1.4.0.post0
nltk==3.7
scikit_learn==0.24.2
wordcloud==1.8.1
torchvision==0.13.1+cu113
requests==2.27.1
tabulate==0.8.9
packaging==21.3
pandas==1.1.5
dgl_cu111==0.6.1
Cython==0.29.28
numpy==1.21.0
FactualSceneGraph
sentence_transformers
SPARQLWrapper==2.0.0
jieba==0.42.1
spacy==3.8.2
```
## 🙏 Acknowledgements
We would like to thank the following researchers for providing the dataset used in this project:
* Xuming Hu (Tsinghua University, Beijing, China)
*  Zhijiang Guo (University of Cambridge, UK)
* Junzhe Chen (Tsinghua University, Beijing, China)
* Lijie Wen (Tsinghua University, Beijing, China)
* Philip S. Yu (University of Illinois at Chicago, USA)

