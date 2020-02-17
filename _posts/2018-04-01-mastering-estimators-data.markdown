---
layout: post
title: "How to solve 90% of NLP tasks in Tensorflow"
description: "Example of tf.estimator and tf.data for Natural Language Processing (named entity recognition)"
excerpt: "An example-based guide to tf.estimator and tf.data for NLP"
date: 2018-03-29
mathjax: true
comments: false
tags: tensorflow NLP
github: https://github.com/guillaumegenthial/estimator_ner
published: true
---

## A paradigm shift towards maturity

## The end of the Session


```python
# We could do : but that does not work for multi class (for now)
precision = tf.metrics.precision(labels=tag_ids, predictions=prediction_ids,
                                 weights=word_mask, name='pre_metrics')
recall = tf.metrics.recall(labels=tag_ids, predictions=prediction_ids,
                           weights=word_mask, name='rec_metrics')
````