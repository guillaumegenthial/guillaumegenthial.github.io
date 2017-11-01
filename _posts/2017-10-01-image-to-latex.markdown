---
layout: post
title:  "Image to LaTeX"
description: "Image to Sequence model with Tensorflow with attention and positional embeddings"
excerpt: "Image to Sequence model with Tensorflow for LaTeX generation"
date:   2017-10-01
mathjax: true
comments: true
published: false
---


Code is available on [github](https://github.com/guillaumegenthial/img2seq).

## Introduction

As an engineering student, how many times did I ask myself

> How amazing would it be if I could take a picture of my math homework and produce a nice LaTeX file out of it?

This thought has been obsessing me for a long time (and I'm sure I'm not the only one) and since I've started studying at Stanford, I've been eager to tackle the problem myself. I must confess that I searched the AppStore for the perfect app, but haven't found anything. I hypothesized that the problem was not that easy and chose to wait until the amazing [computer vision class](http://cs231n.stanford.edu) to tackle the problem.

{% include image.html url="/assets/img2latex/task.svg" description="Producing LaTeX code from an image" size="70%" %}

__The Sequence to Sequence framework__
In my [last post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html), I explained how to predict a tag for a word, which can be considered as a relatively simple task. However, some tasks like translation, require more complicated systems. You may have heard from some recent breakthroughs in Neural Machine Translation that led to (almost) human-level performance systems (used in real-life by Google Translation, see for instance this exciting [work](https://arxiv.org/abs/1611.04558) enabling zero-shot translation). These new architectures all rely on a common paradigm called [__sequence to sequence__](https://arxiv.org/abs/1406.1078) (or __Seq2Seq__), whose goal is to produce an entire sequence of tokens. Compared to former techniques that relied on a translation model (capturing meaning of the input sequence) and a language model (modelling the distribution of words in the output sequence), this framework is more flexible, as it can generate an arbitrary-length sequence after having read the input sequence, while leveraging the flexibility of Deep Learning models (end-to-end training with scalability to any type of input).

> This problem is about producing a sequence of tokens from an image, and is thus at the intersection of Computer Vision and Natural Language Processing.

__Approach__
A similar idea can be applied to our LaTeX generation problem. The input sequence would just be replaced by an image, preprocessed with some convolutional model adapted to OCR (in a sense, if we *unfold* the pixels of an image into a sequence, this is exactly the same problem). This idea proved to be efficient for image captioning (see the famous paper [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)). Building on some [great work](https://arxiv.org/pdf/1609.04938v1.pdf) from the Harvard NLP group, my teammate and I chose to follow a similar approach.

Good Tensorflow implementations of such models were hard to find. Together with this post, I am releasing the [code](https://github.com/guillaumegenthial/img2latex) and hope some will find it useful. You can use it to train your own image captioning model or adapt it for a more advanced use. [The code](https://github.com/guillaumegenthial/img2latex) does __not__ rely on the [Tensorflow Seq2Seq library](https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq) as it was not entirely ready at the time of the project and I also wanted more flexibility. In this post, we'll assume basic knowledge on Deep Learning (Convolutions, LSTMs, etc.). For readers new to Computer Vision and Natural Language Processing, have a look at the wonderful stanford classes [cs231n](http://cs231n.stanford.edu) and [cs224n](http://web.stanford.edu/class/cs224n/).



## The Sequence to Sequence framework

Let's explain the sequence to sequence framework as we'll rely on it for our model. Let's start with the simplest version. Our goal is to translate `how are you` in French `comment ça va`.

### Vanilla Seq2Seq

As explained in the introduction, the Seq2Seq framework usually relies on the __encoder-decoder__ paradigm. The __encoder__ *encodes* the input sequence, while the __decoder__ *produces* the target sequence

__Encoder__

Our input sequence is `how are you`. We'll associate for each word in the sequence a vector $ w \in \mathbb{R}^d $. In our case, we have 3 words, thus our input will be transformed into $ [w_1, w_2, w_3] \in \mathbb{R}^{d \times 3} $. Then, we simply run an LSTM over this sequence of vectors and store the last hidden state outputed by the LSTM: this will be our encoder representation $ h $.

{% include image.html url="/assets/img2latex/seq2seq_vanilla_encoder.svg" description="Vanilla Encoder" size="70%" %}


__Decoder__

Now that we have a vector $ h $ that encapsulates the meaning of the input sequence, we'll use it to generate the target sequence word by word. Feed to another LSTM cell: $ h $ as hidden state and a special *start of sentence* vector $ w_{sos} $ as input. The output of the LSTM will be a vector of the same size of our vocabulary $ s_1 \in \mathbb{R}^{V} $. Let's denote the new hidden state by $ h_1 $.

$$
\begin{align*}
h_1, s_1 &= \operatorname{LSTM}\left(h, w_{sos} \right)\\
p_1 &= \operatorname{softmax}(s_1)\\
i_1 &= \operatorname{argmax}(p_1)\\
\end{align*}
$$

Then, apply a softmax to $ s_1 $ to normalize it into a vector of probabilities $ p_1 \in \mathbb{R}^V $ . Now, each entry of $ p_1 $ will measure how likely is each word in the vocabulary. Let's say that the word *"comment"* has the highest probability (and thus $ i_1 = \operatorname{argmax}(p_1) $ corresponds to the index of *"comment"*). Get a corresponding vector $ w_{i_1} = w_{comment} $ and repeat the procedure: the LSTM will take $ h_1 $ as hidden state and $ w_{comment} $ as input and will output a probability vector $ p_2 $ over the second word, etc.

$$
\begin{align*}
h_2, s_2 &= \operatorname{LSTM}\left(h_1, w_{i_1} \right)\\
p_2 &= \operatorname{softmax}(s_2)\\
\end{align*}$$

The decoding stops when the predicted word is a special *end of sentence* token.

{% include image.html url="/assets/img2latex/seq2seq_vanilla_decoder.svg" description="Vanilla Decoder" size="100%" %}


### Seq2Seq with Global Attention

The previous model has been refined over the past few years and greatly benefited from what is know as __attention__. Attention is a mechanism that forces the model to learn to focus on specific parts of the input sequence when decoding, instead of relying only on the hidden vector of the decoder's LSTM. One way of performing attention is explained in [this paper by Bahdanau](https://arxiv.org/abs/1409.0473).

## Data

To train our model, we'll need labeled examples: images of formulas along with the LaTeX code used to generate the images. A good source of LaTeX code is [arXiv](https://arxiv.org), that has thousands of articles under the `.tex` format. After applying some heuristics to find equations in the `.tex` file, keeping only the ones that actually compile, the [Harvard NLP group](https://zenodo.org/record/56198#.WflVu0yZPLZ) extracted $ \sim 100, 000 $ formulas.

> Wait... Don't you have a problem as different LaTeX codes can give the same equation?

Good point: `(x^2 + 1)` and `\left( x^{2} + 1 \right)` indeed give the same output. That's why Harvard's paper found out that normalizing the data using a parser ([KaTeX](https://khan.github.io/KaTeX/)) improved performance. It forces adoption of some conventions, like writing `x ^ { 2 }` instead of `x^2`, etc. After normalization, they end up with a `.txt` file that looks like

```
\alpha + \beta
\frac { 1 } { 2 }
\frac { \alpha } { \beta }
1 + 2
```

From this file, we'll produce images `0.png`, `1.png`, etc. and a matching file mapping the image files to the indices (=line numbers) of the formulas

```
0.png 0
1.png 1
2.png 2
3.png 3
```

The reason why we use this format is that it is flexible and allows you to use the pre-built [dataset from Harvard](https://zenodo.org/record/56198#.WflVu0yZPLZ). You will need to use the preprocessing script as explained [here](https://github.com/harvardnlp/im2markup). You'll also need to have `pdflatex` and `ImageMagick` installed.

We also build a vocabulary, to map LaTeX tokens to indices that will be given as input to our model. If we keep the same data as above, our vocabulary will look like

`+`
`1`
`2`
`\alpha`
`\beta`
`\frac`
`{`
`}`

## Model

### Encoder


### Decoder


### Attention


### Training


### Beam Search


## Conclusion
