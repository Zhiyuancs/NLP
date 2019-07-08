# Mnemonic Reader
The Mnemonic Reader is a deep learning model for Machine Comprehension task. You can get details from this [paper](https://arxiv.org/pdf/1705.02798.pdf). It combines advantages of [match-LSTM](https://arxiv.org/pdf/1608.07905), [R-Net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) and [Document Reader](https://arxiv.org/abs/1704.00051) and utilizes a new unit, the Semantic Fusion Unit (SFU), to achieve state-of-the-art results (at that time).

This model is a [PyTorch](http://pytorch.org/) implementation of Mnemonic Reader. At the same time, a PyTorch implementation of R-Net and a PyTorch implementation of Document Reader are also included to compare with the Mnemonic Reader. Pretrained models are also available in [release](https://github.com/HKUST-KnowComp/MnemonicReader/releases).

This repo belongs to [HKUST-KnowComp](https://github.com/HKUST-KnowComp) and is under the [BSD LICENSE](LICENSE).

Some codes are implemented based on [DrQA](https://github.com/facebookresearch/DrQA).

Please feel free to contact with Xin Liu (xliucr@connect.ust.hk) if you have any question about this repo.

### Evaluation on SQuAD

| Model                                 | DEV_EM | DEV_F1 |
| ------------------------------------- | ------ | ------ |
| Document Reader (original paper)      | 69.5   | 78.8   |
| Document Reader (trained model)       | 69.6   | 78.6   |
| R-Net (original paper)              | 71.1   | 79.5   |
| R-Net (trained model)                 | 70.7   | 79.7   |
| Mnemonic Reader + RL (original paper) | 72.1   | 81.6   |
| Mnemonic Reader (trained model)       | 72.8   | 81.6   |


### Requirements

* Python >= 3.4
* PyTorch >= 0.31
* spaCy >= 2.0.0
* tqdm
* ujson
* numpy
* prettytable

### Prepare

First of all, you need to download the dataset and pre-trained word vectors.

```bash
mkdir -p data/datasets
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O data/datasets/SQuAD-train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O data/datasets/SQuAD-dev-v1.1.json
```

(you may need `sudo apt install p7zip-full` to install 7z unzip tool)

```bash
mkdir -p data/embeddings
wget https://github.com/csarron/QAModels/releases/download/data/glove.840B.300d.7z -O data/embeddings/glove.840B.300d.7z
cd data/embeddings
7z x glove.840B.300d.7z 
wget https://github.com/csarron/QAModels/releases/download/data/glove.840B.300d-char.txt -O data/embeddings/glove.840B.300d-char.txt

```

Then, you need to preprocess these data.

```bash
python preprocess.py data/datasets data/datasets --split SQuAD-train-v1.1
python preprocess.py data/datasets data/datasets --split SQuAD-dev-v1.1
```

If you want to use multicores to speed up, you could add `--num-workers 4` in commands.

### Train

`python train.py --model-type drqa --model-name drqa --batch-size 32 --dropout-rnn 0.4 2>&1 | tee data/train_drqa.log`

`python train.py --model-type r-net --model-name r-net  --dropout-rnn 0.2 --hidden-size 75  2>&1 | tee data/train_rnet.log`

`python train.py --model-type mnemonic --model-name mnemonic 2>&1 | tee data/train_mnemonic.log`

After several hours, you will get the model in `data/models/`, e.g. `mnemoric.mdl` and you can see the log file in `data/models/`, e.g. `mnemoric.txt`.

### Predict

To evaluate the model you get, you should complete this part.

```bash
python predict.py --model data/models/mnemoric.mdl
```

You need to change the model name in the command above.

You will not get results directly but to use the official `evaluate-v1.1.py` in `data/script`.

```bash
python evaluate-v1.1.py data/predict/SQuAD-dev-v1.1-mnemoric.preds data/datasets/SQuAD-dev-v1.1.json
```

### Interactivate

In order to help those who are interested in QA systems, `interactive.py` provides an easy but good demo.

```bash
python interactive.py --model data/models/mnemoric.mdl
```

Then you will drop into an interactive session. It looks like:

```
* Interactive Module *

* Repo: Mnemonic Reader (https://github.com/HKUST-KnowComp/MnemonicReader)

* Implement based on Facebook's DrQA

>>> process(document, question, candidates=None, top_n=1)
>>> usage()

>>> text="Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
>>> question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
>>> process(text, question)

+------+----------------------------+-----------+
| Rank |            Span            |   Score   |
+------+----------------------------+-----------+
|  1   | Saint Bernadette Soubirous | 0.9875301 |
+------+----------------------------+-----------+
```

### Web Demo

`python demo.py --model data/models/mnemonic-best-epoch_32-em_72.8-f1_81.6.mdl`

### More parameters

If you want to tune parameters to achieve a higher score, you can get instructions about parameters via using

```bash
python preprocess.py --help
```

```bash
python train.py --help
```

```bash
python predict.py --help
```

```bash
python interactive.py --help
```

## License

All codes in **Mnemonic Reader** are under [BSD LICENSE](LICENSE).
