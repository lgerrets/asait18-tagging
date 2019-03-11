
# Audio tagging

## TLDR

Here are some reimplementations to reproduce some experiments from paper [arXiv](http://arxiv.org/abs/1607.03681).

I worked on this as part of a research project at my master MVA at ENS Cachan.

## Contents

Algorithms implemented in this repo make it possible to train an automatic audio multi-tagger in a supervised learning setting. Possible tags include:
- Broadband noise
- Child speech
- Adult female speech
- Adult male speech
- Other identifiable sounds
- Percussive sounds, e.g. crash, bang, knock, footsteps
- Video game/TV

One end-to-end approach is to chain
- **Mel filter banks** (computed in [src/prepare_data.py](src/prepare_data.py))
- with **deep neural nets** (keras architecture [src/models.py](src/models.py) and training procedure [src/train.py](src/train.py)).

This approach is one of those studied by the original paper, which should be read through for better understanding of the above pipeline.


## Installation

In your favorite python environment manager (mine is `conda` right now), install the following python packages:
```
numpy
scipy
librosa (to compute mel filter banks)
keras==2.2.4
tensorflow==1.12.0
```

## Experiments

As previously mentioned, only a part of the experiments from the original paper can be reproduced here, that is:
- tagging audio files with a deep neural net and Mel filter banks.

### Baseline

This repo is intended to be self-sufficient so that experiments can be run without much effort:

First the data needs to be preprocessed:
```
python src/prepare_data.py
```

Then, to train your model:
```
python src/train.py
```

### More data

A small part of the dataset is available in this repo so that experiments can already be run. However that is only 2 chunks when the original dataset actually has 6137. The original dataset can be found [here](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging) along with more information about the challenge and the context.

Once you downloaded the dataset, you can choose to either work with 16kHz or 48kHz sample rates (higher is heavier). If you choose 16kHz, unzip the dataset as follow:
```
tar -xf chime_home.tar.gz --wildcards "chime_home/chunks/*.16kHz.wav" (either 16 or 48)
tar -xf chime_home.tar.gz --wildcards "chime_home/chunks/*.csv"
```

Now, both `src/prepare_data.py` and `src/train.py` take `--data` as a parameter. Just provide the path where the `.wav` and `.csv` files can be found.

### Tweaks

If you want to tweak the codes in [src](src), the comments should be quite complete to understand everything. Please let me know if anything needs clarification.

## Contributions

Contributions are of course welcome!

## Acknowledgements

Original parper:

```
@article{DBLP:journals/corr/XuHWFSJP16,
  author    = {Yong Xu and
               Qiang Huang and
               Wenwu Wang and
               Peter Foster and
               Siddharth Sigtia and
               Philip J. B. Jackson and
               Mark D. Plumbley},
  title     = {Fully Deep Neural Networks Incorporating Unsupervised Feature Learning
               for Audio Tagging},
  journal   = {CoRR},
  volume    = {abs/1607.03681},
  year      = {2016},
  url       = {http://arxiv.org/abs/1607.03681},
  archivePrefix = {arXiv},
  eprint    = {1607.03681},
  timestamp = {Mon, 13 Aug 2018 16:46:06 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/XuHWFSJP16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Some codes are already available at the original research [repo](https://github.com/yongxuUSTC/aDAE_DNN_audio_tagging). However, some codes are missing there to reproduce the full experiemts, which is why I built this repo. Also, the information about the challenge context and the baseline methods are available at this [repo](https://github.com/pafoster/dcase2016_task4).

