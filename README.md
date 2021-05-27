# awesome-speaker-embedding
A curated list of speaker embedding/verification resources

## Code 

| Name |  feature-model-activation/loss | Pre-trained | Datasets|  Link |
| ---- | -------- | -------- | ------- | -------  |
| DeepSpeaker \[1\] |  mel + ResCNN+ Triplet/softmax | Yes |librispeech|  [tensorflow](https://github.com/philipperemy/deep-speaker]) |
|3D CNN\[2\]| MFEC(MFCC)+3DCNN+softmax| No| WVU-Multimodal |  [tensorflow](https://github.com/astorfi/3D-convolutional-speaker-recognition)|
|SincNet\[3\]|raw-wav/sincnet filters+LayerNorm+CNN/DNN+softmax|No|Librispeech|[pytorch](https://github.com/mravanelli/SincNet) [speechbrain](https://github.com/speechbrain/speechbrain)|
|VGGVox\[4\]|mel+VGG/Resnet+softmax/Pair-wise contrastive|Yes|voxceleb1&2 | [Matconvnet](https://github.com/a-nagrani/VGGVox) |
|GE2E\[5\]|Log-fbank + LSTM + GE2E-loss|Yes| Ok-Google| [pytorch](https://github.com/HarryVolek/PyTorch_Speaker_Verification),[Tensorflow](https://github.com/Janghyun1230/Speaker_Verification) |

## Papers
- \[1\] [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/abs/1705.02304), Baidu inc, 2017
- \[2\] [Text-Independent Speaker Verification Using 3D Convolutional Neural Networks](https://arxiv.org/abs/1705.09422), 2017
- \[3\] [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158), Bengio team,  raw waveform, 2018
- \[4\] [VoxCeleb2: Deep Speaker Recognition](https://arxiv.org/abs/1806.05622) VGG group, Interspeech 2018
- \[5\] [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467), Google, ICASSP 2017
- \[6\] [Voxceleb: Large-scale speaker verification in the wild]()https://www.robots.ox.ac.uk/~vgg/publications/2019/Nagrani19/nagrani19.pdf, VGG, 2019
- \[7\] [Deep neural network embeddings for text-independent speaker verification](http://danielpovey.com/files/2017_interspeech_embeddings.pdf), Interspeech 2017, original TDNN paper from Johns Hopkins , use MFCC, frame-based, time-delay, multi-class, softmax + cross-entropy loss
- \[8\] [Robust DNN Embeddings for Speaker Recognition](https://arxiv.org/pdf/1803.09153v1.pdf), ICASSP 2018, the <b>X-vector</b> paper Johns Hopkins,  based on TDNN, improved by adding Noise and reverberation for augmentation
- \[9\] [Front-end factor analysis for speaker verification](http://groups.csail.mit.edu/sls/archives/root/publications/2010/Dehak_IEEE_Transactions.pdf), 2011, IEEE TASLP,  the '<b>i-vector</b>' paper from Johns Hopkins 
- \[10\] [TDNN-UBM Time delay deep recognition neural network-based universal background models for speaker](https://www.danielpovey.com/files/2015_asru_tdnn_ubm.pdf) , 2015 
- \[11\] [Deep neural networks for small footprint text-dependent speaker verification](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf), The '<b>D-vector</b>' paper from Johns Hopkins 

### More-recent papers
- [Attention Back-end](https://arxiv.org/pdf/2104.01541.pdf), Compare PLDA and cosine with proposed attention Back-end, model: TDNN, Resnet, data: cn-celeb



## Other resources
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), high-level representation of a voice through a deep learning model (referred to as the voice encoder).
- [voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube

- [NetVLAD](https://github.com/lyakaap/NetVLAD-pytorch)
- [Triplet-loss](https://omoindrot.github.io/triplet-loss)

## Benchmarks
- Voxceleb1 \[6\]
![vox1](./res/vox1_benchmark.png)