---
layout: post
title:  "Intro to tf.estimator and tf.data"
description: "Introduction to tf.estimator and tf.data for Natural Language Processing (NLP)"
excerpt: "An example for Natural Language Processing (NER)"
date:   2018-11-18
mathjax: true
comments: true
tags: tensorflow NLP
github: https://github.com/guillaumegenthial/tf_ner
published: False
---


Code is available on [github](https://github.com/guillaumegenthial/tf_ner/blob/master/models/lstm_crf/main.py).

## Outline

<!-- MarkdownTOC -->

- Introduction
- Global configuration
- Feed data to your model with tf.data
    - A simple example
    - Test your tf.data pipeline
    - Read from file and tokenize
- Defining our model with tf.estimator

<!-- /MarkdownTOC -->



## Introduction

Not so long ago, designing Deep Learning software meant writing custom `Model` classes. However, with code and implementations piling up on github and other open-source platforms, developers soon had enough examples to start designing unified high-level APIs for models and data loading. After all, we are all lazy and writing boilerplate code is not necessarily our cup of tea.

Thanks to intense competition on the Deep Learning Framework landscape, driving innovation and development of new functionnalities, brillant minds at Google and elsewhere worked on new additions to Tensorflow. Don't ask me which version has it all (but I'm pretty sure 1.9 is good enough).

For those unfamiliar with Tensorflow, just wanting to bootstrap their project and reading on the web good and bad things about it, especially how so much more user-friendly its rival, Facebook's newborn [pyTorch](https://pytorch.org/) is, well, forget about those sterile debates and try them both: they are more than good enough and when it comes to chosing which one to use for your company or your project, it depends more on what existing implementation is available and your technical debt than the frameworks' specificities.

This blog post provides an __example__ and gives some advice about how to use the new high-level Tensorflow APIs and demonstrate how you can quickly get a state-of-the-art NLP model up and running with as few as a 100 lines of code!

## Global configuration

If you like having an idea about what's happening when you're running your code, you need to change the __level of verbosity__ of tensorflow logging (the module `tf.logging` which is built on top of the standard `logging` module). It does seem a bit weird that we have to take care of it, but hey, nothing is perfect and there probably is a reason that I am unaware of (nobody's omniscient).

This snippet of code does a bit more than changing the verbosity: it also writes everything that tensorflow logs to a __file__. This is especially useful when you execute your code on a cluster of machines thanks to the help of a job manager (like slurm). The logs of `stdout` are probably accessible somewhere else, but I still think it's better to have a version of it along with your weights and other stuff.

```python
import logging
from pathlib import Path
import sys

import tensorflow as tf

# Setup logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers
```

> If you don't know / don't use the `pathlib` module (python3 only), try using it. It bundles a lot of `os.path` functionnality (and more) in a much nicer and easy-to-use package. I started using it after reading about it on [this blog](https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f) which also has a lot of other excellent articles.

> We create two `handlers`: one that will write the logs to `sys.stdout` (the terminal window so that you can see what's going on), and one to a file (as the `FileHandler` name implies). I prefer resetting the `handlers` completely to avoid double-logging.

So far, no Deep Learning, no Tensorflow (or barely), just python trickery.


## Feed data to your model with tf.data


When it comes to feeding data to your model (and to the tensorflow graph), there has been a few possible ways of achieving this in the past. The standard technique was to use a `tf.placeholder` that was updated through the `run` method of a `tf.Session` object. There was also an attempt of a more optimized input pipeline with [threadings and queues](https://www.tensorflow.org/api_guides/python/threading_and_queues).

Forget all this (first of all because we won't use `tf.Session` anymore).

A better (and almost perfect) way of feeding data to your tensorflow model is to use a wonderful new tensorflow API called [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) (thanks to the efforts of [Derek Murray](https://github.com/mrry) and others) whose philosophy, in a few words, is to create a special node of the graph that knows how to iterate the data and yield batches of tensors.

There is a bunch of official tutorials that you can find on the [official website](https://www.tensorflow.org/) - something alas quite unusual. Here is a (short) selection:

- [The official guide - Importing Data](https://www.tensorflow.org/guide/datasets)
- [How to use `tf.data.Dataset` with the high level model API `tf.estimator`](https://www.tensorflow.org/guide/datasets_for_estimators)
- [The Neural Machine Translation Tutorial - A good example for NLP](https://github.com/tensorflow/nmt)

### A simple example

I am a great fan of the flexibility provided by `tf.data.Dataset.from_generator`. It allows you to do the data loading (from file or elsewhere) and some preprocessing in python before feeding it into the graph. Basically, defining such a dataset is just wiring the outputs of any python generator into Tensorflow Tensors.


Let's define a dummy data generator

```python
def generator_fn():
    for digit in range(2):
        line = 'I am digit {}'.format(digit)
        words = line.split()
        yield [w.encode() for w in words], len(words)
```

which yields
```python
for words in generator_fn():
    print(words)
>>> ([b'I', b'am', b'digit', b'0'], 4)
>>> ([b'I', b'am', b'digit', b'1'], 4)
```
> __Good to know__: when feeding string objects to your graph, you need to encode your string to `bytes`.


Now, we want to make the output of this generator available inside our graph. Let's create a special `Dataset` node


```python
shapes = ([None], ())
types = (tf.string, tf.int32)

dataset = tf.data.Dataset.from_generator(generator_fn,
    output_shapes=shapes, output_types=types)
```

### Test your tf.data pipeline


> It is __necessary to test__ that your `tf.data` pipeline works as expected to avoid long hours of unnecessary headache and debugging.

There are 2 techniques to test your `dataset`.

1. Use the new `eager_execution` mode

    ```python
    import tensorflow as tf
    tf.enable_eager_execution()

    for tf_words, tf_size in dataset:
        print(tf_words, tf_size)
    >>> tf.Tensor([b'I' b'am' b'digit' b'0'], shape=(4,), dtype=string) tf.Tensor(4, shape=(), dtype=int32)
    >>> tf.Tensor([b'I' b'am' b'digit' b'1'], shape=(4,), dtype=string) tf.Tensor(4, shape=(), dtype=int32)
    ```
    > The `tf.enable_eager_execution()` must be called at program startup, just after your `import tensorflow as tf`

2. Use an old-school, not-so-user-friendly-but-still-usefull `tf.Session`. Before that, we need to create an `iterator` out of our dataset. In other words, a dataset is an object that can be iterated, but we need to get the node created when iterating the dataset in order to evaluate it explicitely via a `Session`.

    ```python
    iterator = dataset.make_one_shot_iterator()
    node = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run(node))
        print(sess.run(node))  # Each call moves the iterator to its next position
    >>> (array([b'I', b'am', b'digit', b'0'], dtype=object), 4)
    >>> (array([b'I', b'am', b'digit', b'1'], dtype=object), 4)
    ```
    > `print(node)` sums up the above explanation in *tensorflow-ic* jargon - though a bit abstruse `(<tf.Tensor 'IteratorGetNext_1:0' shape=(?,) dtype=string>, <tf.Tensor 'IteratorGetNext_1:1' shape=() dtype=int32>)`

### Read from file and tokenize

Now, a more complete example. We have 2 files `words` and `tags`, each line containing a white-spaced tokenized tagged sentence.

1. The generator function, that reads the files and parse the lines

    ```python
    def parse_fn(line_words, line_tags):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags), "Words and tags lengths don't match"
        return (words, len(words)), tags


    def generator_fn(words, tags):
        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield parse_fn(line_words, line_tags)
    ```
    > Again, notice how we encode strings into `bytes`

2. The `input_fn` that constructs the dataset (we will need this function to work with `tf.estimator` later on)

    ```python
    def input_fn(words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')

        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset

    ```
    > Notice how we perform a few operations on our `dataset` like `shuffle`, `repeat`, `padded_batch` and `prefetch`. These operations are self-explanatory, except maybe for `prefetch` which ensures that a batch of data is pre-loaded on the computing device so that it does not suffer from data starvation (= wasting compute resources that have to wait for the data to be transfered, a problem that occurs with GPU and a low compute / data ratios)

We don't forget to try out our new pipeline. Let's create 2 files
1. `words.txt`
    ```
    I live in San Francisco
    You live in Paris
    ```

2. `tags.txt`
    ```
    O O O B-LOC I-LOC
    O O O S-LOC
    ```

And run the following code

```python
dataset = input_fn('words.txt', 'tags.txt')
iterator = dataset.make_one_shot_iterator()
node = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run(node))
>>> ((array([[b'I', b'live', b'in', b'San', b'Francisco'],
>>>          [b'You', b'live', b'in', b'Paris', b'<pad>']],
>>>         dtype=object),
>>>   array([5, 4], dtype=int32)),
>>>   array([[b'O', b'O', b'O', b'B-LOC', b'I-LOC'],
>>>          [b'O', b'O', b'O', b'S-LOC', b'O']],
>>>          dtype=object))
```
> As you can see, the `padded_batch` worked as desired, by adding new tokens to the shorter sentence. Now, we have a batch of data which is ready to be consumed by our model!


## Defining our model with tf.estimator

