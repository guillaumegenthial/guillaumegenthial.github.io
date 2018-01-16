---
layout: post
title:  "Seq2Seq with Attention and Beam Search"
description: "Sequence to Sequence basics for Neural Machine Translation using Attention and Beam Search"
excerpt: "Seq2Seq for LaTeX generation - part I"
date:   2017-11-08
mathjax: true
comments: true
published: true
tags: NLP Vision
---

This post is the first in a series about [__im2latex__](https://guillaumegenthial.github.io/image-to-latex.html): its goal is to cover the __concepts__ of Sequence-to-Sequence models with Attention and Beam search.

> If you're already familiar with Seq2Seq and want to go straight to the Tensorflow code

<div align="right" > <a href="https://guillaumegenthial.github.io/image-to-latex.html"><h3>> Go to part II</h3></a>  </div>


## Introduction

In my [last post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html) about named entity recognition, I explained how to predict a tag for a word, which can be considered as a relatively simple task. However, some tasks like translation require more complicated systems. You may have heard from some recent breakthroughs in Neural Machine Translation that led to (almost) human-level performance systems (used in real-life by Google Translation, see for instance this [paper](https://arxiv.org/abs/1611.04558) enabling zero-shot translation). These new architectures rely on a common paradigm called __encoder-decoder__ (or __sequence to sequence__), whose goal is to produce an entire sequence of tokens.

In this post, we'll assume basic knowledge about Deep Learning (Convolutions, LSTMs, etc.). For readers new to Computer Vision and Natural Language Processing, have a look at the famous Stanford classes [cs231n](http://cs231n.stanford.edu) and [cs224n](http://web.stanford.edu/class/cs224n/).


## Sequence to Sequence basics

Let's explain the sequence to sequence framework as we'll rely on it for our model. Let's start with the simplest version on the translation task.

> As an example, let's translate `how are you` in French `comment vas tu`.

### Vanilla Seq2Seq

The Seq2Seq framework relies on the __encoder-decoder__ paradigm. The __encoder__ *encodes* the input sequence, while the __decoder__ *produces* the target sequence

__Encoder__

Our input sequence is `how are you`. Each word from the input sequence is associated to a vector $ w \in \mathbb{R}^d $ (via a lookup table). In our case, we have 3 words, thus our input will be transformed into $ [w_0, w_1, w_2] \in \mathbb{R}^{d \times 3} $. Then, we simply run an LSTM over this sequence of vectors and store the last hidden state outputed by the LSTM: this will be our encoder representation $ e $. Let's write the hidden states $ [e_0, e_1, e_2] $ (and thus $ e = e_2 $)

{% include image.html url="/assets/img2latex/seq2seq_vanilla_encoder.svg" description="Vanilla Encoder" size="60%" %}


__Decoder__

Now that we have a vector $ e $ that captures the meaning of the input sequence, we'll use it to generate the target sequence word by word. Feed to another LSTM cell: $ e $ as hidden state and a special *start of sentence* vector $ w_{sos} $ as input. The LSTM computes the next hidden state $ h_0 \in \mathbb{R}^h $. Then, we apply some function $ g : \mathbb{R}^h \mapsto \mathbb{R}^V $ so that $ s_0 := g(h_0) \in \mathbb{R}^V $ is a vector of the same size as the vocabulary.

$$
\begin{align*}
h_0 &= \operatorname{LSTM}\left(e, w_{sos} \right)\\
s_0 &= g(h_0)\\
p_0 &= \operatorname{softmax}(s_0)\\
i_0 &= \operatorname{argmax}(p_0)\\
\end{align*}
$$

Then, apply a softmax to $ s_0 $ to normalize it into a vector of probabilities $ p_0 \in \mathbb{R}^V $ . Now, each entry of $ p_0 $ will measure how likely is each word in the vocabulary. Let's say that the word *"comment"* has the highest probability (and thus $ i_0 = \operatorname{argmax}(p_0) $ corresponds to the index of *"comment"*). Get a corresponding vector $ w_{i_0} = w_{comment} $ and repeat the procedure: the LSTM will take $ h_0 $ as hidden state and $ w_{comment} $ as input and will output a probability vector $ p_1 $ over the second word, etc.

$$
\begin{align*}
h_1 &= \operatorname{LSTM}\left(h_0, w_{i_0} \right)\\
s_1 &= g(h_1)\\
p_1 &= \operatorname{softmax}(s_1)\\
i_1 &= \operatorname{argmax}(p_1)
\end{align*}$$

The decoding stops when the predicted word is a special *end of sentence* token.

{% include image.html url="/assets/img2latex/seq2seq_vanilla_decoder.svg" description="Vanilla Decoder" size="90%" %}

> Intuitively, the hidden vector represents the "amount of meaning" that has not been decoded yet.

The above method aims at modelling the distribution of the next word conditionned on the beginning of the sentence

$$ \mathbb{P}\left[ y_{t+1} | y_1, \dots, y_{t}, x_0, \dots, x_n \right] $$

by writing

$$ \mathbb{P}\left[ y_{t+1} | y_t, h_{t}, e \right] $$

### Seq2Seq with Attention

The previous model has been refined over the past few years and greatly benefited from what is known as __attention__. Attention is a mechanism that forces the model to learn to focus (=to attend) on specific parts of the input sequence when decoding, instead of relying only on the hidden vector of the decoder's LSTM. One way of performing attention is explained by [Bahdanau et al.](https://arxiv.org/abs/1409.0473). We slightly modify the reccurrence formula that we defined above by adding a new vector $ c_t $ to the input of the LSTM


$$
\begin{align*}
h_{t} &= \operatorname{LSTM}\left(h_{t-1}, [w_{i_{t-1}}, c_t] \right)\\
s_t &= g(h_t)\\
p_t &= \operatorname{softmax}(s_t)\\
i_t &= \operatorname{argmax}(p_t)
\end{align*}$$

The vector $ c_t $ is the attention (or __context__) vector. We compute a new context vector at each decoding step. First, with a function $ f (h_{t-1}, e_{t'}) \mapsto \alpha_{t'} \in \mathbb{R} $, compute a score for each hidden state $ e_{t'} $ of the encoder. Then, normalize the sequence of $ \alpha_{t'} $ using a softmax and compute $ c_t $ as the weighted average of the $ e_{t'} $.

$$
\begin{align*}
\alpha_{t'} &= f(h_{t-1}, e_{t'})  \in \mathbb{R} & \text{for all } t'\\
\bar{\alpha} &= \operatorname{softmax} (\alpha)\\
c_t &= \sum_{t'=0}^n \bar{\alpha}_{t'} e_{t'}
\end{align*}
$$

{% include image.html url="/assets/img2latex/seq2seq_attention_mechanism_new.svg" description="Attention Mechanism" size="100%" %}


The choice of the function $ f $ varies, but is usually one of the following

$$
f(h_{t-1}, e_{t'}) =
\begin{cases}
h_{t-1}^T e_{t'} & \text{dot}\\
h_{t-1}^T W e_{t'} & \text{general}\\
v^T \tanh \left(W [h_{t-1}, e_{t'}]\right) & \text{concat}\\
\end{cases}
$$

It turns out that the attention weighs $ \bar{\alpha} $ can be easily interpreted. When generating the word `vas` (corresponding to `are` in English), we expect $ \bar{\alpha} _ {\text{are}} $ to be close to $ 1 $ while $ \bar{\alpha} _ {\text{how}} $ and $ \bar{\alpha} _ {\text{you}} $ to be close to $ 0 $. Intuitively, the context vector $ c $ will be roughly equal to the hidden vector of `are` and it will help to generate the French word `vas`.

By putting the attention weights into a matrix (rows = input sequence, columns = output sequence), we would have access to the __alignment__ between the words from the English and French sentences... (see [page 6](https://arxiv.org/pdf/1409.0473.pdf)) There is still a lot of things to say about sequence to sequence models (for instance, it works better if the encoder processes the input sequence *backwards*...).


## Training

> What happens if the first time step is not sure about wether it should generate `comment` or `vas` (most likely case at the beginning of the training)? Then it would mess up the entire sequence, and the model will hardly learn anything...

If we use the predicted token as input to the next step during training (as explained above), errors would accumulate and the model would rarely be exposed to the correct distribution of inputs, making training slow or impossible. To speedup things, one trick is to feed the actual output sequence (`<sos>` `comment` `vas` `tu`) into the decoder's LSTM and predict the next token at every position (`comment` `vas` `tu` `<eos>`).

{% include image.html url="/assets/img2latex/img2latex_training.svg" description="Training" size="80%" %}


The decoder outputs vectors of probability over the vocabulary $ p_i \in \mathbb{R}^V  $ for each time step. Then, for a given target sequence $ y_1, \dots, y_n $, we can compute its probability as the product of the probabilities of each token being produced at each relevant time step:

$$
\mathbb{P}\left(y_1, \dots, y_m \right) = \prod_{i=1}^m p_i [y_i]
$$

where $  p_i [y_i] $ means that we extract the $ y_i $-th entry of the probability vector $ p_i $ from the $i$-th decoding step. In particular, we can compute the probability of the actual target sequence. A perfect system would give a probabilty of 1 to this target sequence, so we are going to train our network to maximize the probability of the target sequence, which is the same as minimizing

$$
\begin{align*}
-\log \mathbb{P} \left(y_1, \dots, y_m \right) &= - \log \prod_{i=1}^m p_i [y_i]\\
&= - \sum_{i=1}^n \log p_i [y_i]\\
\end{align*}
$$

in our example, this is equal to

$$ - \log p_1[\text{comment}] - \log p_2[\text{vas}] - \log p_3[\text{tu}] - \log p_4[\text{<eos>}] $$


and you recognize the standard cross entropy: we actually are minimizing the cross entropy between the target distribution (all one-hot vectors) and the predicted distribution outputed by our model (our vectors $ p_i $).


## Decoding

The main takeaway from the discussion above is that for the same model, we can define different behaviors. In particular, we defined a specific behavior that speeds up training.

> What about inference/testing time then? Is there an other way to decode a sentence?

There indeed are 2 main ways of performing decoding at testing time (translating a sentence for which we don't have a translation). The first of these methods is the one covered at the beginning of the article: __greedy decoding__. It is the most natural way and it consists in feeding to the next step the most likely word predicted at the previous step.

{% include image.html url="/assets/img2latex/seq2seq_vanilla_decoder.svg" description="Greedy Decoder - feeds the best token to the next step" size="70%" %}


> But didn't we say that this behavior is likely to accumulate errors?

Even after having trained the model, it can happen that the model makes a small error (and gives a small advantage to `vas` over `comment` for the first step of the decoding). This would mess up the entire decoding...

There is a better way of performing decoding, called __Beam Search__. Instead of only predicting the token with the best score, we keep track of $ k $ hypotheses (for example $ k = 5 $, we refer to $ k $ as the __beam size__). At each new time step, for these 5 hypotheses we have $ V $ new possible tokens. It makes a total of $ 5 V $ new hypotheses. Then, only keep the $ 5 $ best ones, and so on... Formally, define $$ \mathcal{H}_ t $$ the set of hypotheses decoded at time step $$ t $$.

$$ \mathcal{H}_ t := \{ (w^1_1, \dots, w^1_t), \dots, (w^k_1, \dots, w^k_t) \} $$

For instance if $$ k = 2 $$, one possible $$ \mathcal{H}_ 2 $$ would be


$$ \mathcal{H}_ 2 := \{ (\text{comment vas}), (\text{comment tu}) \} $$


Now we consider all the possible candidates $$ \mathcal{C}_ {t+1}$$, produced from $$ \mathcal{H}_ t $$ by adding all possible new tokens

$$ \mathcal{C}_ {t+1} := \bigcup_{i=1}^k \{ (w^i_1, \dots, w^i_t, 1), \dots, (w^i_1, \dots, w^i_t, V) \} $$

and keep the $$ k $$ highest scores (probability of the sequence). If we keep our example

$$ \begin{align*}
\mathcal{C}_ 3 =& \{ (\text{comment vas comment}), (\text{comment vas vas}), (\text{comment vas tu})\}  \\
\cup & \{ (\text{comment tu comment}), \ \ (\text{comment tu vas}), \ \ (\text{comment tu tu}) \}
\end{align*}
$$

and for instance we can imagine that the 2 best ones would be

$$ \mathcal{H}_ 3 := \{ (\text{comment vas tu}), (\text{comment tu vas}) \} $$


Once every hypothesis reached the `<eos>` token, we return the hypothesis with the highest score.

> If we use __beam search__, a small error at the first step might be rectified at the next step, as we keep the gold hypthesis in the beam!


## Conclusion


In this article we covered the seq2seq concepts. We showed that training is different than decoding. We covered two methods for decoding: __greedy__ and __beam search__. While beam search generally achieves better results, it is not perfect and still suffers from __exposure bias__. During training, the model is never exposed to its errors! It also suffers from __Loss-Evaluation Mismatch__. The model is optimized w.r.t. token-level cross entropy, while we are interested about the reconstruction of the whole sentence...


Now, let's apply Seq2Seq for LaTeX generation from images!

<div align="right" > <a href="https://guillaumegenthial.github.io/image-to-latex.html"><h2>> Go to part II</h2></a>  </div>


__Classic papers about seq2seq__

- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)
- [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

__More advanced papers trying to address some limitations__
- [An Actor-Critic Algorithm for sequence prediction](https://arxiv.org/pdf/1607.07086.pdf)
- [Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/pdf/1606.02960.pdf)
- [Six Challenges for Neural Machine Translation](https://arxiv.org/pdf/1706.03872.pdf)
- [Professor Forcing: A New Algorithm for Training Recurrent Networks](https://arxiv.org/abs/1610.09038)