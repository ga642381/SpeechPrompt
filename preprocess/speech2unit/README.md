# Speech to Unit Model (speech2unit)

## Speech Prompt

This Speech2unit/ is cloned from the original fairseq GitHub repo.
The only modified file is **Speech2unit/clustering/quantize_with_kmeans**

## Acoustic Model

For quantizing speech we learn a K-means clustering over acoustic representations for which we either use Log-Mel Filterbank or pretrained acoustic representation models. For using pretrained models, please download from their respective locations linked below.

- [Modified CPC](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/cpc_big_ll6kh_top_ctc.pt)
- [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
- [Wav2Vec 2.0-Base](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt)

## Quantization Model

You can download pretrained quantized model from the list below.

| K-Means Model              | Download Link                                                                    |
| -------------------------- | -------------------------------------------------------------------------------- |
| Log Mel Filterbank + KM50  | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km50/km.bin)  |
| Log Mel Filterbank + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km100/km.bin) |
| Log Mel Filterbank + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km200/km.bin) |
| Modified CPC + KM50        | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km50/km.bin)     |
| Modified CPC + KM100       | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km100/km.bin)    |
| Modified CPC + KM200       | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km200/km.bin)    |
| HuBERT Base + KM50         | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin)  |
| HuBERT Base + KM100        | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin) |
| HuBERT Base + KM200        | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin) |
| wav2vec 2.0 Large + KM50   | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km50/km.bin)    |
| wav2vec 2.0 Large + KM100  | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km100/km.bin)   |
| wav2vec 2.0 Large + KM200  | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km200/km.bin)   |
