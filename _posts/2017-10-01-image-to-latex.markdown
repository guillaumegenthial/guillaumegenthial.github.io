---
layout: post
title:  "Seq2seq for LaTeX generation"
description: "Sequence to Sequence model (seq2seq) in Tensorflow + attention + positional embeddings + beam search - Im2LaTeX challenge - similar to Show Attend and Tell"
excerpt: "Sequence to Sequence model with Tensorflow for LaTeX generation"
date:   2017-11-08
mathjax: true
comments: true
published: true
---


Code is available on [github](https://github.com/guillaumegenthial/im2latex).
Though designed for *image to LaTeX* ([im2latex](https://openai.com/requests-for-research/#im2latex) challenge), it could be used for standard seq2seq with very little effort.

## Introduction

As an engineering student, how many times did I ask myself

> How amazing would it be if I could take a picture of my math homework and produce a nice LaTeX file out of it?

This thought has been obsessing me for a long time (and I'm sure I'm not the only one) and since I've started studying at Stanford, I've been eager to tackle the problem myself. I must confess that I searched the AppStore for the perfect app, but haven't found anything. I hypothesized that the problem was not that easy and chose to wait until the amazing [computer vision class](http://cs231n.stanford.edu) to tackle the problem.

{% include image.html url="/assets/img2latex/img2latex_task.svg" description="Producing LaTeX code from an image" size="100%" %}

__The Sequence to Sequence framework__
In my [last post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html) about named entity recognition, I explained how to predict a tag for a word, which can be considered as a relatively simple task. However, some tasks like translation require more complicated systems. You may have heard from some recent breakthroughs in Neural Machine Translation that led to (almost) human-level performance systems (used in real-life by Google Translation, see for instance this exciting [work](https://arxiv.org/abs/1611.04558) enabling zero-shot translation). These new architectures rely on a common paradigm called [__sequence to sequence__](https://arxiv.org/abs/1406.1078) (or __Seq2Seq__), whose goal is to produce an entire sequence of tokens. <!-- Compared to former techniques that relied on a translation model (capturing meaning of the input sequence) and a language model (modeling the distribution of words in the output sequence), this framework is more flexible, as it can generate an arbitrary-length sequence after having read the input sequence, while leveraging the flexibility of Deep Learning models (end-to-end training with scalability to any type of input).
 -->
> This problem is about producing a sequence of tokens from an image, and is thus at the intersection of Computer Vision and Natural Language Processing.

__Approach__
A similar idea can be applied to our LaTeX generation problem. The input sequence would just be replaced by an image, preprocessed with some convolutional model adapted to OCR (in a sense, if we *unfold* the pixels of an image into a sequence, this is exactly the same problem). This idea proved to be efficient for image captioning (see the reference paper [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)). Building on some [great work](https://arxiv.org/pdf/1609.04938v1.pdf) from the Harvard NLP group, my teammate [Romain](https://www.linkedin.com/in/romain-sauvestre-241171a2) and I chose to follow a similar approach.

Good Tensorflow implementations of such models were hard to find. Together with this post, I am releasing the [code](https://github.com/guillaumegenthial/im2latex) and hope some will find it useful. You can use it to train your own image captioning model or adapt it for a more advanced use. [The code](https://github.com/guillaumegenthial/im2latex) does __not__ rely on the [Tensorflow Seq2Seq library](https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq) as it was not entirely ready at the time of the project and I also wanted more flexibility (but adopts a similar interface). In this post, we'll assume basic knowledge about Deep Learning (Convolutions, LSTMs, etc.). For readers new to Computer Vision and Natural Language Processing, have a look at the famous Stanford classes [cs231n](http://cs231n.stanford.edu) and [cs224n](http://web.stanford.edu/class/cs224n/).



## Sequence to Sequence basics

Let's explain the sequence to sequence framework as we'll rely on it for our model. Let's start with the simplest version on the tranlation task. As an example, let's translate `how are you` in French `comment vas tu`.

### Vanilla Seq2Seq

The Seq2Seq framework relies on the __encoder-decoder__ paradigm. The __encoder__ *encodes* the input sequence, while the __decoder__ *produces* the target sequence

__Encoder__

Our input sequence is `how are you`. Each word from the input sequence is associated to a vector $ w \in \mathbb{R}^d $ (via a lookup table). In our case, we have 3 words, thus our input will be transformed into $ [w_0, w_1, w_2] \in \mathbb{R}^{d \times 3} $. Then, we simply run an LSTM over this sequence of vectors and store the last hidden state outputed by the LSTM: this will be our encoder representation $ e $. Let's write the hidden states $ [e_0, e_1, e_2] $ (and thus $ e = e_2 $)

{% include image.html url="/assets/img2latex/seq2seq_vanilla_encoder.svg" description="Vanilla Encoder" size="70%" %}


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

{% include image.html url="/assets/img2latex/seq2seq_attention_mechanism.svg" description="Attention Mechanism" size="100%" %}


The choice of the function $ f $ varies, but is usually one of the following

$$
f(h_{t-1}, e_{t'}) =
\begin{cases}
h_{t-1}^T e_{t'}\\
h_{t-1}^T W e_{t'}\\
w^T [h_{t-1}, e_{t'}]\\
\end{cases}
$$

It turns out that the attention weighs $ \bar{\alpha} $ can be easily interpreted. When generating the word `vas` (corresponding to `are` in English), we expect $ \bar{\alpha} _ {\text{are}} $ to be close to $ 1 $ while $ \bar{\alpha} _ {\text{how}} $ and $ \bar{\alpha} _ {\text{you}} $ to be close to $ 0 $. Intuitively, the context vector $ c $ will be roughly equal to the hidden vector of `are` and it will help to generate the French word `vas`.

By putting the attention weights into a matrix (rows = input sequence, columns = output sequence), we would have access to the __alignment__ between the words from the English and French sentences... (see [page 6](https://arxiv.org/pdf/1409.0473.pdf)) There is still a lot of things to say about sequence to sequence models (for instance, it works better if the encoder processes the input sequence *backwards*...) but we know enough to get started on our LaTeX generation problem.


## Data

To train our model, we'll need labeled examples: images of formulas along with the LaTeX code used to generate the images. A good source of LaTeX code is [arXiv](https://arxiv.org), that has thousands of articles under the `.tex` format. After applying some heuristics to find equations in the `.tex` files, keeping only the ones that actually compile, the [Harvard NLP group](https://zenodo.org/record/56198#.WflVu0yZPLZ) extracted $ \sim 100, 000 $ formulas.

> Wait... Don't you have a problem as different LaTeX codes can give the same image?

Good point: `(x^2 + 1)` and `\left( x^{2} + 1 \right)` indeed give the same output. That's why Harvard's paper found out that normalizing the data using a parser ([KaTeX](https://khan.github.io/KaTeX/)) improved performance. It forces adoption of some conventions, like writing `x ^ { 2 }` instead of `x^2`, etc. After normalization, they end up with a `.txt` file containing one formula per line that looks like

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

The reason why we use this format is that it is flexible and allows you to use the pre-built [dataset from Harvard](https://zenodo.org/record/56198#.WflVu0yZPLZ) (You may need to use the preprocessing scripts as explained [here](https://github.com/harvardnlp/im2markup)). You'll also need to have `pdflatex` and `ImageMagick` installed.

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

Our model is going to rely on a variation of the Seq2Seq model, adapted to images. First, let's define the input of our graph. Not surprisingly we get as input a batch of black-and-white images of shape $ [H, W] $ and a batch of formulas (ids of the LaTeX tokens):

```python
# batch of images, shape = (batch size, height, width, 1)
img = tf.placeholder(tf.uint8, shape=(None, None, None, 1), name='img')
# batch of formulas, shape = (batch size, length of the formula)
formula = tf.placeholder(tf.int32, shape=(None, None), name='formula')
# for padding
formula_length = tf.placeholder(tf.int32, shape=(None, ), name='formula_length')
```


> A special note on the type of the image input. You may have noticed that we use `tf.uint8`. This is because our image is encoded in grey-levels (integers from `0` to `255` - and $ 2^8 = 256 $). Even if we could give a `tf.float32` Tensor as input to Tensorflow, this would be 4 times more expensive in terms of memory bandwith. As data starvation is one of the main bottlenecks of GPUs, this simple trick can save us some training time. For further improvement of the data pipeline, have a look at [the new Tensorflow data pipeline](https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/data).

### Encoder

__High-level idea__ Apply some convolutional network on top of the image an flatten the output into a sequence of vectors $ [e_1, \dots, e_n] $, each of those corresponding to a region of the input image. These vectors will correspond to the hidden vectors of the LSTM that we used for translation.

> Once our image is transformed into a sequence, we can use the seq2seq model!

{% include image.html url="/assets/img2latex/img2latex_encoder.svg" description="Convolutional Encoder - produces a sequence of vectors" size="100%" %}


We need to extract features from our image, and for this, nothing has ([yet](https://arxiv.org/abs/1710.09829)) been proven more effective than convolutions. Here, there is nothing much to say except that we pick some architecture that has been proven to be effective for Optical Character Recognition (OCR), which stacks convolutional layers and max-pooling to produce a Tensor of shape $ [H', W', 512] $


```python
# casting the image back to float32 on the GPU
img = tf.cast(img, tf.float32) / 255.

out = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)
out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)

out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)
out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

# encoder representation, shape = (batch size, height', width', 512)
out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)
```

Now that we have extracted some features from the image, let's __unfold__ the image to get a sequence so that we can use our sequence to sequence framework. We end up with a sequence of length $ [H' \times W'] $

```python
H, W = tf.shape(out)[1:2]
seq = tf.reshape(out, shape=[-1, H*W, 512])
```


> Don't you loose a lot of structural information by reshaping? I'm afraid that when performing attention over the image, my decoder won't be able to understand the location of each feature vector in the original image!

It turns out that the model manages to work despite this issue, but that's not completely satisfying. In the case of translation, the hidden states of the LSTM contained some positional information that was computed by the LSTM (after all, LSTM are by essence sequential). Can we fix this issue?

__Positional Embeddings__ I decided to follow the idea from [Attention is All you Need](https://arxiv.org/abs/1706.03762) that adds *positional embeddings* to the image representation (`out`), and has the huge advantage of not adding any new trainable parameter to our model. The idea is that for each position of the image, we compute a vector of size $ 512 $ such that its components are $ \cos $ or $ \sin $. More formally, the (2i)-th and (2i+1)-th entries of my positional embedding $ v $ at position $ p $ will be

$$
\begin{align*}
v_{2i} &= \sin\left(p / f^{2i}\right)\\
v_{2i+1} &= \cos\left(p / f^{2i}\right)\\
\end{align*}
$$

where $ f $ is some frequency parameter. Intuitively, because $ \sin(a+b) $ and $ \cos(a+b) $ can be expressed in terms of $ \sin(b)$ , $ \sin(a)$ , $ \cos(b)$  and $ \cos (a) $, there will be linear dependencies between the components of distant embeddings, authorizing the model to extract relative positioning information. Good news: the tensorflow code for this technique is available in the library [tensor2tensor](https://github.com/tensorflow/tensor2tensor), so we just need to reuse the same function and transform our `out` with the following call

```python
out = add_timing_signal_nd(out)
```

<!-- > Harvard's technique is slightly different, as they process the feature vectors with a recurrent neural network, that adds positional information (among other things). -->

### Decoder

Now that we have a sequence of vectors $ [e_1, \dots, e_n] $ that represents our input image, let's decode it! First, let's explain what variant of the Seq2Seq framework we are going to use.

__First hidden vector of the decoder's LSTM__ In the seq2seq framework, this is usually just the last hidden vector of the encoder's LSTM. Here, we don't have such a vector, so a good choice would be to learn to compute it with a matrix $ W $ and a vector $ b $

$$
h_0 = \tanh\left( W \cdot \left( \frac{1}{n} \sum_{i=1}^n e_i\right) + b \right)
$$

This can be done in Tensorflow with the following logic

```python
img_mean = tf.reduce_mean(seq, axis=1)
W = tf.get_variable("W", shape=[512, 512])
b = tf.get_variable("b", shape=[512])
h = tf.tanh(tf.matmul(img_mean, W) + b)
```


__Attention Mechanism__ We first need to compute a score $ \alpha_{t'} $ for each vector $ e_{t'} $ of the sequence. We use the following method

$$
\begin{align*}
\alpha_{t'} &= \beta^T \tanh\left( W_1 \cdot e_{t'} + W_2 \cdot h_{t}  \right)\\
\bar{\alpha} &= \operatorname{softmax}\left(\alpha\right)\\
c_t &= \sum_{i=1}^n \bar{\alpha}_{t'} e_{t'}\\
\end{align*}
$$

This can be done in Tensorflow with the follwing code

```python
# over the image, shape = (batch size, n, 512)
W1_e = tf.layers.dense(inputs=seq, units=512, use_bias=False)
# over the hidden vector, shape = (batch size, 512)
W2_h = tf.layers.dense(inputs=h, units=512, use_bias=False)

# sums the two contributions
a = tf.tanh(W1_e + tf.expand_dims(W2_h, axis=1))
beta = tf.get_variable("beta", shape=[512, 1], dtype=tf.float32)
a_flat = tf.reshape(a, shape=[-1, 512])
a_flat = tf.matmul(a_flat, beta)
a = tf.reshape(a, shape=[-1, n])

# compute weights
a = tf.nn.softmax(a)
a = tf.expand_dims(a, axis=-1)
c = tf.reduce_sum(a * seq, axis=1)
```

> Note that the line `W1_e = tf.layers.dense(inputs=seq, units=512, use_bias=False)` is common to every decoder time step, so we can just compute it once and for all. The dense layer with no bias is just a matrix multiplication.

Now that we have our attention vector, let's just add a small modification and compute an other vector $ o_{t-1} $ (as in [Luong, Pham and Manning](https://arxiv.org/pdf/1508.04025.pdf)) that we will use to make our final prediction and that we will feed as input to the LSTM at the next step. Here $ w_{t-1} $ denotes the embedding of the token generated at the previous step.

>  $$ o_t $$ passes some information about the distribution from the previous time step as well as the confidence it had for the predicted token

$$
\begin{align*}
h_t &= \operatorname{LSTM}\left( h_{t-1}, [w_{t-1}, o_{t-1}] \right)\\
c_t &= \operatorname{Attention}([e_1, \dots, e_n], h_t)\\
o_{t} &= \tanh\left(W_3 \cdot [h_{t}, c_t] \right)\\
p_t &= \operatorname{softmax}\left(W_4 \cdot o_{t} \right)\\
\end{align*}
$$

and now the code

```python
# compute o
W3_o = tf.layers.dense(inputs=tf.concat([h, c], axis=-1), units=512, use_bias=False)
o = tf.tanh(W3_o)

# compute the logits scores (before softmax)
logits = tf.layers.dense(inputs=o, units=vocab_size, use_bias=False)
# the softmax will be computed in the loss or somewhere else
```


> If I read carefully, I notice that for the first step of the decoding process, we need to compute an $ o_0 $ too, right?

This is a good point, and we just use the same technique that we used to generate $ h_0 $ but with different weights.

## Tensorflow details

> Well, now that we've covered the essential steps, how to I make it work with the high level functions of Tensorflow like `dynamic_rnn` etc. ?

We'll need to encapsulate the reccurent logic into a custom cell that inherits `RNNCell`. Our custom cell will be able to call the LSTM cell (initialized in the `__init__`). It also has a special recurrent state that combines the LSTM state and the vector $ o $ (as we need to pass it through). An elegant way is to define a namedtuple for this recurrent state:

```python
AttentionState = collections.namedtuple("AttentionState", ("lstm_state", "o"))

class AttentionCell(RNNCell):
    def __init__(self):
        self.lstm_cell = LSTMCell(512)

    def __call__(self, inputs, cell_state):
        """
        Args:
            inputs: shape = (batch_size, dim_embeddings) embeddings from previous time step
            cell_state: (AttentionState) state from previous time step
        """
        lstm_state, o = cell_state
        # compute h
        h, new_lstm_state = self.lstm_cell(tf.concat([inputs, o], axis=-1), lstm_state)
        # apply previous logic
        c = ...
        new_o  = ...
        logits = ...

        new_state = AttentionState(new_lstm_state, new_o)
        return logits, new_state
```


Then, to compute our output sequence, we just need to call the previous cell on the sequence of LaTeX tokens. We first produce the sequence of token embeddings to which we concatenate the special `<sos>` token. Then, we call `dynamic_rnn`.

```python
# 1. get token embeddings
E = tf.get_variable("E", shape=[vocab_size, 80], dtype=tf.float32)
# special <sos> token
start_token = tf.get_variable("start_token", dtype=tf.float32, shape=[80])
tok_embeddings = tf.nn.embedding_lookup(E, formula)

# 2. add the special <sos> token embedding at the beggining of every formula
start_token_ = tf.reshape(start_token, [1, 1, dim])
start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
# remove the <eos> that won't be used because we reached the end
tok_embeddings = tf.concat([start_tokens, tok_embeddings[:, :-1, :]], axis=1)

# 3. decode
attn_cell = AttentionCell()
seq_logits, _ = tf.nn.dynamic_rnn(attn_cell, tok_embeddings, initial_state=AttentionState(h_0, o_0))
```


> If I understand the code written above, you are feeding the `formula` into the decoder's LSTM! In other words, you're not generating any sentence, but merely copying!

That's partly right! The previous code does indeed feed into the decoder's LSTM the actual tokens of the target sequence, and thus is trained to predict the next word at each position. It will speedup the training, as errors won't accumulate. If the decoder is wrong about the first token, which is the most likely case at the beginning of the training, then, if we feed this token to the next step, the second token will have even smaller chances to be correct!. That's why we replace the prediction we feed into the LSTM by the actual true token at training time.


> We'll need to create 2 different outputs in the Tensorflow graph: one for training (that uses the `formula`) and one for test time (that ignores everything about the actual `formula` and uses the prediction from the previous step).


## Training

During training, we feed the actual output sequence (`<sos>` `comment` `vas` `tu`) into the decoder's LSTM and it tries to predict the next token at every position (`comment` `vas` `tu` `<eos>`).

{% include image.html url="/assets/img2latex/img2latex_training.svg" description="Training" size="80%" %}


 The `__call__` function of our `AttentionCell` outputs `logits`, a vector of scores $ \in \mathbb{R}^V $ for each time step. After applying a softmax to each of these logit vectors, we get vectors of probability over the vocabulary $ p_i \in \mathbb{R}^V  $ for each time step. Then, for a given target sequence $ y_1, \dots, y_n $, we can compute its probability as the product of the probabilities of each token being produced at each relevant time step:

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


and you recognize the standard cross entropy: we actually are minimizing the cross entropy between the target distribution (all one-hot vectors) and the predicted distribution outputed by our model (our vectors $ p_i $). Now, thanks to high-level functions of Tensorflow, the implementation is pretty straightforward, we only need to be careful about the masking (for formulas that have different length in the batch).



```python
# compute - log(p_i[y_i]) for each time step, shape = (batch_size, formula length)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seq_logits, labels=formula)
# masking the losses
mask = tf.sequence_mask(formula_length)
losses = tf.boolean_mask(losses, mask)
# averaging the loss over the batch
loss = tf.reduce_mean(losses)
# building the train op
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
```

and when iterating over the batches during training, `train_op` will be given to the `tf.Session` along with a `feed_dict` containing the data for the placeholders.

## Testing and Decoding in Tensorflow

The natural way of performing decoding is the one explained at the beginning of the article: choose the token that has the highest probability at each time step, before feeding it to the next time step. This method is called __greedy decoding__ and works pretty well, but suffers from the fact that it makes local choices. That's why we'll also cover another method that explores more options and can avoid local traps, the __beam search__.

> Let's have a look at the Tensorflow implementation of the greedy method first

### Greedy Decoder

{% include image.html url="/assets/img2latex/seq2seq_vanilla_decoder.svg" description="Greedy Decoder - feeds the best token to the next step" size="70%" %}


While greedy decoding is easy to conceptualize, implementing it in Tensorflow is not straightforward, as you need to use the previous prediction and can't use `dynamic_rnn` on the `formula`. There are basically __2 ways of approaching the problem__

1. Modify our `AttentionCell` and `AttentionState` so that `AttentionState` also contains the embedding of the predicted word at the previous time step,
```python
    AttentionState = namedtuple("AttentionState", ("lstm_state", "o", "embedding"))

    class AttentionCell(RNNCell):
        def __call__(self, inputs, cell_state):
            lstm_state, o, embbeding = cell_state
            # compute h
            h, new_lstm_state = self.lstm_cell(tf.concat([embedding, o], axis=-1), lstm_state)
            # usual logic
            logits = ...
            # compute new embeddding
            new_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)
            new_state = AttentionState(new_lstm_state, new_o, new_embedding)

            return logits, new_state
```
> This technique has a few downsides. It __doesn't use `inputs`__ (which used to be the embedding of the gold token from the `formula` and thus we would have to call `dynamic_rnn` on a "fake" sequence). Also, how do you know when to stop decoding, once you've reached the `<eos>` token?

2. Implement a variant of `dynamic_rnn` that would not run on a sequence but feed the prediction from the previous time step to the cell, while having a maximum number of decoding steps. This would involve delving deeper into Tensorflow, using `tf.while_loop`. That's the method we're going to use as it fixes all the problems of the first technique. We eventually want something that looks like
```python
attn_cell = AttentionCell(...)
# wrap the attention cell for decoding
decoder_cell = GreedyDecoderCell(attn_cell)
# call a special dynamic_decode primitive
test_outputs, _ = dynamic_decode(decoder_cell, max_length_formula+1)
```
> Much better isn't it? Now let's see what `GreedyDecoderCell` and `dynamic_decode` look like.

### Greedy Decoder Cell
We first wrap the attention cell in a `GreedyDecoderCell` that takes care of the greedy logic for us, without having to modify the `AttentionCell`

```python
class DecoderOutput(collections.namedtuple("DecoderOutput", ("logits", "ids"))):
    pass

class GreedyDecoderCell(object):
    def step(self, time, state, embedding, finished):
        # next step of attention cell
        logits, new_state = self._attention_cell.step(embedding, state)
        # get ids of words predicted and get embedding
        new_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)
        # create new state of decoder
        new_output = DecoderOutput(logits, new_ids)
        new_finished = tf.logical_or(finished, tf.equal(new_ids,
                self._end_token))

        return (new_output, new_state, new_embedding, new_finished)

```


### Dynamic Decode primitive

We need to implement a function `dynamic_decode` that will recursively call the above `step` function. We do this with a `tf.while_loop` that stops when all the hypotheses reached `<eos>` or `time` is greater than the max number of iterations.

```python
def dynamic_decode(decoder_cell, maximum_iterations):
    # initialize variables (details on github)

    def condition(time, unused_outputs_ta, unused_state, unused_inputs, finished):
        return tf.logical_not(tf.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished):
        new_output, new_state, new_inputs, new_finished = decoder_cell.step(
            time, state, inputs, finished)
        # store the outputs in TensorArrays (details on github)
        new_finished = tf.logical_or(tf.greater_equal(time, maximum_iterations), new_finished)

        return (time + 1, outputs_ta, new_state, new_inputs, new_finished)

    with tf.variable_scope("rnn"):
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_state, initial_inputs, initial_finished])

    # return the final outputs (details on github)
```
> Some details using `TensorArrays` or `nest.map_structure` have been omitted for clarity but may be found on [github](https://github.com/guillaumegenthial/im2latex/blob/master/model/components/dynamic_decode.py)

> Notice that we place the `tf.while_loop` inside a scope named `rnn`. This is because `dynamic_rnn` does the same thing and thus the weights of our LSTM are defined in that scope.



### Beam Search

> What happens if the first time step is not sure about wether it should generate `comment` or `vas`? Imagine they have very close scores, but `vas` is slightly higher: then it would mess up the entire decoding! Is there a way to fix it?

There indeed is a better way of performing decoding, called __beam search__. Instead of only predicting the token with the best score, we keep track of $ k $ hypotheses (for example $ k = 5 $, we refer to $ k $ as the __beam size__). At each new time step, for these 5 hypotheses we have $ V $ new possible tokens. It makes a total of $ 5 V $ new hypotheses. Then, only keep the $ 5 $ best ones, and so on... Formally, define $$ \mathcal{H}_ t $$ the set of hypotheses decoded at time step $$ t $$.

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

### Beam Search Decoder Cell

> We can follow the same approach as in the greedy method and use `dynamic_decode`

Let's create a new wrapper for `AttentionCell` in the same way we did for `GreedyDecoderCell`. This time, the code is going to be more complicated and the following is just for intuition. Note that when selecting the top $$ k $$ hypotheses from the set of candidates, we must know which "beginning" they used (=parent hypothesis).

```python
class BeamSearchDecoderCell(object):

    # notice the same arguments as for GreedyDecoderCell
    def step(self, time, state, embedding, finished):
        # compute new logits
        logits, new_cell_state = self._attention_cell.step(embedding, state.cell_state)

        # compute log probs of the step (- log p(w) for all words w)
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = tf.nn.log_softmax(new_logits)

        # compute scores for the (beam_size * vocabulary_size) new hypotheses
        log_probs = state.log_probs + step_log_probs

        # get top k hypotheses
        new_probs, indices = tf.nn.top_k(log_probs, self._beam_size)

        # get ids of next token along with the parent hypothesis
        new_ids = ...
        new_parents = ...

        # compute new embeddings, new_finished, new_cell state...
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)
```
> Look at [github](https://github.com/guillaumegenthial/im2latex/blob/master/model/components/beam_search_decoder_cell.py) for the details. The main idea is that we add a beam dimension to every tensor, but when feeding it into `AttentionCell` we merge the beam dimension with the batch dimension. There is also some trickery involved to compute the parents and the new ids using modulos.

## Conclusion

__Limitations__ In this article we covered the seq2seq framework in details, from the concepts to the code. We also applied it to the [im2latex](https://openai.com/requests-for-research/#im2latex) challenge. We showed that training is different than decoding. We covered two methods for decoding: __greedy__ and __beam search__. While beam search generally achieves better results, it is not perfect and still suffers from __exposure bias__. During training, the model is never exposed to its errors! It also suffers from __Loss-Evaluation Mismatch__. During training, the model is optimized w.r.t. token-level cross entropy, while we are interested about the reconstruction of the whole sentence.

__Metrics__ This raises the question of *how do we evaluate the performance of our model?*. We can use standard metrics from Machine Translation like [BLEU](https://en.wikipedia.org/wiki/BLEU) to evaluate how good the decoded LaTeX is compared to the reference. We can also choose to compile the predicted LaTeX sequence to get the image of the formula, and then compare this image to the orignal. As a formula is a sequence, computing the pixel-wise distance wouldn't really make sense. A good idea is proposed by [Harvard's paper](http://lstm.seas.harvard.edu/latex). First, slice the image vertically. Then, compare the edit distance between these slices...

__Tensorflow__ At some point, we omitted details about `TensorArrays` or `nest.map_structure`. It turned out to be easier than I thought to use these low level elements of Tensorflow. The `nest` module was particularly useful for applying a function to a structure of Tensors (like our `AttentionState`). The `TensorArrays` could not be circumvented for the `while_loop` but caused little problems. The use of these tools makes recurrent logic programming possible and dimishes the significance of some critics made to Tensorflow.

{% include double-image.html
    url1="/assets/img2latex/ref.png" caption1=""
    url2="/assets/img2latex/pred.png" caption2=""
    size="50%"
    description="Error example - which one is the reference?" %}
