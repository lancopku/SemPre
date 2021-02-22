# SemPre

Code for *Towards Semantics-Enhanced Pre-Training: Can Lexicon Definitions Help Learning Sentence Meanings?* (AAAI 2021)

---

Self-supervised pre-training techniques, albeit relying on
large amounts of text, have enabled rapid growth in learning
language representations for natural language understanding.
However, as radically empirical models on sentences, they
are subject to the input data distribution, inevitably incorporating
data bias and reporting bias, which may lead to inaccurate
understanding of sentences. To address this problem,
we propose to adopt a human learner’s approach: when we
cannot make sense of a word in a sentence, we often consult
the dictionary for specific meanings; but can the same
work for empirical models? In this work, we try to inform
the pre-trained masked language models of word meanings
for semantics-enhanced pre-training. To achieve a contrastive
and holistic view of word meanings, a definition pair of two
related words is presented to the masked language model such
that the model can better associate a word with its crucial semantic
features. Both intrinsic and extrinsic evaluations validate
the proposed approach on semantics-orientated tasks,
with an almost negligible increase of training data.

A preliminary version of the paper can be found [here](docs/paper.pdf). The appendix can be found [here](docs/appendix.pdf).

#### Predicting Definitional Knowledge

<div align="center"><img src="docs/fig-ma.png" height="120px" alt="Predicting Definitional Knowledge"/></div>


#### Propagating Definitional Knowledge through Semantic Relations

<div align="center"><img src="docs/fig-mb.png" height="120px" alt="Propagating Definitional Knowledge through Semantic Relations"/></div>

#### Better Learned Word Representation

<div style="display: flex; justify-content: space-around">
  <div display="inline-block" align="center">
    <img src="docs/fig-vis-roberta.png" height="200px" alt="RoBERTa-large"/>
    <br>
    <span>RoBERTa-large</span>
  </div>
  <div display="inline-block" align="center">
   <img src="docs/fig-vis-sempre.png" height="200px" alt="RoBERTa-large with SemPre"/>
    <br>
    <span>RoBERTa-large with SemPre</span>
  </div>
</div>


---

## Usage

#### Requirements

- python 3.6
- pytorch 1.5.1
- fairseq 0.9.0
- nltk

We runned the experiments on NVIDIA GPUs. `conda` is recommended. NOTE: the codes have been re-organized quite a bit. Especially, we monkey patch fairseq to backport some functionalities. If you encounter problems, please contact the authors.

#### Dataset

For training, we extract wordnet definitions using nltk and you may need preprocesssed bookwiki for validation, which is optional though. For validation, we use [WiC](https://pilehvar.github.io/wic/), [PIQA](https://yonatanbisk.com/piqa/), [CATs](https://github.com/XuhuiZhou/CATS), which can be obtained from the corresponding links. WGs can be processed using scripts at [here](scripts/utils/preprocess_data.py).

#### Training

Running SemPre on RoBERTa-large with 4 Titan RTX:
```
MODEL=roberta.large/model  # path for downloaded .pt checkpoint
DATA_ROOT=data-bin         # where the data is
DATA=bookcorpus_all        # fairseq-preprocessed data, not actually used in training
                           # a folder with dict.txt, label_dict.txt, valid.idx and
                           # valid.bin should be fine

cd scripts

GPU=0,1,2,3 NUM_GPU=4 ARGS="--fp16 --fp16-init-scale 1" MODEL=$MODEL DATA_ROOT=$DATA_ROOT DATA=$DATA ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=128 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256 USE_LAMB=1 bash run-sempre_full-roberta.sh
```

RoBERTa-base should run fine with 1 2080 Ti. For other usages, please see `run-sempre_{full|partial|simple}.sh`.

#### Validation

The instructions are provided in the corresponding scripts:
- WGs: [`word_guess.py`](scripts/word_guess.py)
- CATs: [`making_sense.py`](scripts/making_sense.py)
- WiC: [`run-wic.sh`](scripts/run-wic.sh)
- PIQA: [`run-piqa.sh`](scripts/run-piqa.sh)


#### Checkpoints

Trained RoBERTa-base and RoBERTa-large with SemPre can be found [here](https://drive.google.com/drive/folders/1aDfSQMwZ_tBAk2XEiBaYzd6Ebqfvj2gt?usp=sharing). Best-performing validation logs for WiC and PIQA can also found there.

### Citation

If you find the paper and the code helpful to your research, please cite as:
```
@inproceedings{ren2021towards,
  author    = {Xuancheng Ren and
               Xu Sun and
               Houfeng Wang and
               Qun Liu},
  title     = {Towards Semantics-Enhanced Pre-Training: Can Lexicon Definitions
               Help Learning Sentence Meanings?},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2021}
}
```