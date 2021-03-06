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

#### Environment Setup

- Ubuntu 18.04
- python 3.6
- pytorch 1.3.1 + cudatoolkit 10.0
- fairseq 0.9.0
- nltk
- NVIDIA apex

We run the experiments on NVIDIA GPUs. `conda` is recommended. NOTE: the codes have been re-organized quite a bit. Especially, we monkey patch fairseq to backport some functionalities. If you encounter problems, please contact the authors.

To replicate the results:
```
conda create -n sempre python=3.6 pytorch=1.3.1 cudatoolkit=10.0 -c pytorch
conda activate sempre
conda install nltk
# some of fairseq's dependencies can be installed by conda:
# conda install requests cython future portalocker=2.0

# install NVIDIA apex using the pre-complied version
pip install ./scripts/utils/apex-0.1-cp36-cp36m-linux_x86_64.whl
pip install fairseq==0.9.0
```


#### Dataset

For training, we extract wordnet definitions using nltk and you may need other data for MLM validation, which is optional though. If you have never used WordNet related functions in NLTK, you need to download it first. It will be prompt by nltk, please follow the given instructions or:
```
conda activate sempre
python -c "import nltk; nltk.download('wordnet')"
```

For validation, we use [WiC](https://pilehvar.github.io/wic/), [PIQA](https://yonatanbisk.com/piqa/), [CATs](https://github.com/XuhuiZhou/CATS), which can be obtained from the corresponding links. WGs can be processed using scripts at [here](scripts/utils/preprocess_data.py).

#### Training

The given scripts supposed using conda with an environment named `sempre`. 

We use pre-trained RoBERTa-{base,large} which can be downloaded [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md). Please extract the content from the archive and place the model checkpoint model.pt under `scripts/checkpoints/roberta.{base,large}`.


Running SemPre on RoBERTa-large with 4 Titan RTX:
```
cd scripts

MODEL=roberta.large/model  # path for downloaded .pt checkpoint

DATA_ROOT=data-bin/         # where the data is

DATA=data                  # a folder with dict.txt (required), label_dict.txt (required),
                           # valid.bin (optional), and valid.idx (optional)
                           # valid.bin and valid.idx is needed only if MLM validation is needed,
                           # and they should be preprocessed using GPT2-BPE and dict.txt

# tune file-max for multi-GPU training
# ulimit -n 4096

# using the following command, the checkpoints will be named as $MODEL/batchsize2048-seed1234/checkpoint{1...10}.pt
GPU=0,1,2,3 NUM_GPU=4 ARGS="--fp16 --fp16-init-scale 1" MODEL=$MODEL DATA_ROOT=$DATA_ROOT DATA=$DATA ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=128 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256 USE_LAMB=1 bash run-sempre_full-roberta.sh
```

RoBERTa-base should run fine with 1 2080 Ti. For other usages, please see `run-sempre_{full|partial|simple}.sh`.

#### Validation

The instructions are provided in the corresponding scripts. To replicate the results, please use the SemPre checkpoints provided below. Place the checkpoints as `scripts/checkpoints/roberta.{base,large}/model/batchsize2048-seed1234/checkpoint1.pt`:
- WGs: [`word_guess.py`](scripts/word_guess.py)
- CATs: [`making_sense.py`](scripts/making_sense.py)
- WiC: [`run-wic.sh`](scripts/run-wic.sh)  
  Please download the data in jsonl from [here](https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip). Place the jsonl files under `scripts/data-raw/WiC`. To replicate the results, run the following with a 2080Ti GPU.
  ```
  cd scripts
  
  MODEL=checkpoints/roberta.large/model/batchsize2048-seed1234/checkpoint1
  DATA_ROOT=data-raw/WiC/
  
  # the script defaults to the best hyperparameters and will run validation and conduct
  # inference on test. The best checkpoint, logs, and the test results will be 
  # written to a subfolder named WiC-batchsize32-lr1e-5-seed5-me50
  GPU=0 ARGS="--fp16 --fp16-init-scale 1" MODEL=$MODEL bash run-wic.sh
  ```
- PIQA: [`run-piqa.sh`](scripts/run-piqa.sh)
  Please download the data from [here](https://yonatanbisk.com/piqa/data/) (merge the .lst to the .jsonl as a field named label). Place the jsonl files under `scripts/data-raw/PIQA`. To replicate the results, run the following with a 2080Ti GPU.
  ```
  cd scripts
  
  MODEL=checkpoints/roberta.large/model/batchsize2048-seed1234/checkpoint1
  DATA_ROOT=data-raw/PIQA/
  
  # the script defaults to the best hyperparameters and will run validation and conduct
  # inference on test. The best checkpoint, logs, and the test results will be 
  # written to a subfolder named PIQA-batchsize32-lr1e-5-seed9100-me50
  GPU=0 ARGS="--fp16 --fp16-init-scale 1" MODEL=$MODEL bash run-piqa.sh
  ```


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