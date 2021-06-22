# awesome-speaker-embedding
A curated list of speaker embedding/verification resources


## Must-read papers
- \[01\] [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/abs/1705.02304), Baidu inc, 2017
- \[02\] [Text-Independent Speaker Verification Using 3D Convolutional Neural Networks](https://arxiv.org/abs/1705.09422), 2017
- \[03\] [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158), Bengio team,  raw waveform, 2018
- \[04\] [VoxCeleb2: Deep Speaker Recognition](https://arxiv.org/abs/1806.05622) VGG group, Interspeech 2018
- \[05\] [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467), Google, ICASSP 2017
- \[06\] [Voxceleb: Large-scale speaker verification in the wild](https://www.robots.ox.ac.uk/~vgg/publications/2019/Nagrani19/nagrani19.pdf),VGG group, 2019
- \[07\] [Deep neural network embeddings for text-independent speaker verification](http://danielpovey.com/files/2017_interspeech_embeddings.pdf), Interspeech 2017, original <b>TDNN</b> paper from Johns Hopkins , MFCC/frame-based/time-delay/multi-class, softmax + cross-entropy loss
- \[08\] [Robust DNN Embeddings for Speaker Recognition](https://arxiv.org/pdf/1803.09153v1.pdf), ICASSP 2018, the <b>X-vector</b> paper Johns Hopkins,  based on TDNN, improved by adding Noise and reverberation for augmentation
- \[09\] [Front-end factor analysis for speaker verification](http://groups.csail.mit.edu/sls/archives/root/publications/2010/Dehak_IEEE_Transactions.pdf), 2011, IEEE TASLP,  the '<b>i-vector</b>' paper from Johns Hopkins 
- \[10\] [TDNN-UBM Time delay deep recognition neural network-based universal background models for speaker](https://www.danielpovey.com/files/2015_asru_tdnn_ubm.pdf) , 2015 
- \[11\] [Deep neural networks for small footprint text-dependent speaker verification](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf), The '<b>D-vector</b>' paper from Johns Hopkins 
- \[12\] [Analysis of Score Normalization in Multilingual Speaker Recognition](http://www.fit.vutbr.cz/research/groups/speech/publi/2017/matejka_interspeech2017_IS170803.pdf), Interspeech 2017, The S-norm paper, useful for score normalization 


## Benchmarks (Voxceleb1)

Results reported (by the authors) on [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt), [VoxCeleb1-E](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt) and [VoxCeleb1-H](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt).

Voxceleb1 public leaderboard (continuously updating...)
| Name |  feature,model,activation/loss |  VoxCeleb1| VoxCeleb1-E| VoxCeleb1-H| Link |Affiliation|Year |
| ---- | -------- | -------- | ------- | -------  |-------  |-------  |--------  |
|X205| DPN68,Res2Net50| 0.7712%| 0.8968%| 1.637% |[report](https://arxiv.org/pdf/2011.00200.pdf) | AISpeech | 2020|
|Veridas| ResNet152|1.08%|-|-|[report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/veridas.pdf)|das-nano|2020
|DKU-DukeECE | Resnet,ECAPA-TDNN| 0.888%|1.133%|2.008%|[report](https://arxiv.org/pdf/2010.12731.pdf)|Duke University|2020|
|IDLAB | Resnet,ECAPA-TDNN| 2.1%|-|-|[report](https://arxiv.org/pdf/2010.12468.pdf)|Ghent University -|2020|

## Must-read technical reports

[VOXSRC 2019 reports](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/VoxSRC19.pdf)

## Datasets
Commonly-used speaker datasets: 
- [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1): A small dataset for speaker and asr, non-free
- [Free ST](https://www.openslr.org/38/): Mandarin speech corpus for speaker and asr, free 
- [NIST SRE](https://sre.nist.gov/) NIST Speaker Recognition Evaluation, non-free
- [AIShell-1](https://www.openslr.org/33/): Mandarin speech corpus, divided into train/dev/test, free. 
- [AIShell-2](http://www.aishelltech.com/aishell_2): free for education, non-free for commercial
- [AIShell-3](https://www.openslr.org/93/): free, for speaker, asr and tts
- [AIShell-4](https://arxiv.org/abs/2104.03603), will be released soon
- [HI-MIA](https://www.openslr.org/85/): free, for far-field text-dependent  speaker verification and  keyword spotting
- [SITW](http://www.speech.sri.com/projects/sitw/) Speakers in the Wild, 
- [Voxceleb 1&2](https://www.openslr.org/82/), Celebrity interview video/audio extracted from Youtube
- [Cn-Celeb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), Multi-genres speaker dataset in the wild, utterances are from chinese celebrities. 

## Challenges
- [VoxCeleb Speaker Recognition Challenge (VoxSRC 2019)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2019.html) [report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/VoxSRC19.pdf)
- [VoxCeleb Speaker Recognition Challenge (VoxSRC 2020)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2020.html)
- [VoxCeleb Speaker Recognition Challenge (VoxSRC 2021)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2021.html)
- [Short-duration Speaker Verification (SdSV) Challenge 2020](https://sdsvc.github.io/2020/)
- [Short-duration Speaker Verification (SdSV) Challenge 2021](https://sdsvc.github.io/)
- [CTS Speaker Recognition Challenge 2020](https://sre.nist.gov/cts-challenge)
- [Far-Field Speaker Verification Challenge (FFSVC 2020)](http://2020.ffsvc.org/)

## Great Talks / Tutorials
- [X-vectors: Neural Speech Embeddings for Speaker Recognition](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/keynote/daniel_talk.mp4), Daniel Garcia-Romero, 2020
- [2020声纹识别研究与应用学术讨论会](https://hub.baai.ac.cn/view/4289)



## Code 

| Name |  feature-model-activation/loss | Pre-trained | Datasets|  Link |
| ---- | -------- | -------- | ------- | -------  |
| DeepSpeaker \[1\] |  FBank + ResCNN+ Triplet/softmax | Yes |librispeech|  [tensorflow](https://github.com/philipperemy/deep-speaker]) |
|3D CNN\[2\]| MFEC(MFCC)+3DCNN+softmax| No| WVU-Multimodal |  [tensorflow](https://github.com/astorfi/3D-convolutional-speaker-recognition)|
|SincNet\[3\]|raw-wav/sincnet filters+LayerNorm+CNN/DNN+softmax|No|Librispeech|[pytorch](https://github.com/mravanelli/SincNet) [speechbrain](https://github.com/speechbrain/speechbrain)|
|VGGVox\[4\]|mel+VGG/Resnet+softmax/Pair-wise contrastive|Yes|voxceleb1&2 | [Matconvnet](https://github.com/a-nagrani/VGGVox) |
|GE2E\[5\]|Log-fbank + LSTM + GE2E-loss|Yes| Ok-Google| [pytorch](https://github.com/HarryVolek/PyTorch_Speaker_Verification),[Tensorflow](https://github.com/Janghyun1230/Speaker_Verification) |

## Tools/Frameworks/libraries
- [asv-subtools](https://github.com/Snowdar/asv-subtools)  An Open Source Tools based on Pytorch and Kaldi for speaker recognition/language identification, XMU Speech Lab. 
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), high-level representation of a voice through a deep learning model (referred to as the voice encoder).
- [voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube
- [Triplet-loss](https://omoindrot.github.io/triplet-loss) Triplet Loss and Online Triplet Mining in TensorFlow. 
- [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels) The Res2net architecture used commonly in VoxCeleb speaker recognition challenge. 
- [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) A very good speaker framework written in pytorch with pretrained models. 
- [Speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)  Voxceleb recipe. 
- [kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb) Kaldi recipe for voxceleb. 
- [pytorch_xvectors](https://github.com/manojpamk/pytorch_xvectors) pytorch implementation of x-vectors. 
### More-recent papers
- [Attention Back-end](https://arxiv.org/pdf/2104.01541.pdf), Compare PLDA and cosine with proposed attention Back-end, model: TDNN, Resnet, data: cn-celeb


### Wining solutions of Completions
#### VoxSRC2019
- Rank 1:  FBank, "r-vectors" using resnet, AAM loss. From Brno University of Technolog, [REPORT](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop/BUT_Zeinali_VoxSRC.pdf)
- Rank 2: 80-dim FBank features, E-TDNN/F-TDNN models, various classification loss including softmax/AM-softmax/PLDA-softmax. From Johns Hopkins University, [REPORT](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop/JHU-HLTCOE_VoxSRC.pdf)
- Rank 3: FBank, resnet + attentive pooling + Phonetic attention, BLSTM + ResNET, loss unclear(?). From Microsoft, [REPORT](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop/VoxSRC_TZ_microsoft.pdf)


#### VoxSRC2020
- Rank 1: 60-dim log-FBank, ECAPA-TDNN/SE-ResNet34, S-Norm, AAM-Softmax. From IDLab, [REPORT](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/participants/JTBD.pdf)
- Rank 2: 40-dim FBank/mean-normalized, no VAD, resnet/Res2Net, S-Norm, CM-Softmax. From AI Speech, [REPORT](https://arxiv.org/pdf/2011.00200.pdf), kaldi [recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb) for data-aug
- Rank 3: Report not available




