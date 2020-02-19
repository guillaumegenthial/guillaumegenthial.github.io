---
layout: post
title: "Machine Learning is Software Engineering"
description: "Machine Learning Engineering in the Industry"
excerpt: "What I learned during my first year on the job"
mathjax: true
comments: true
tags: ML Industry
published: False
---


*Palaiseau, France, some rainy day of 2015. Early morning.* As I'm walking to class, I run into one of my closest friends. For the past few months, we've been feeling the need to find a new goal, a new spark, something worth pursuing on the longer term. We've been studying theoretical maths and physics for a few years now, and sure, the classes here become more and more interesting from an intellectual point of view ; the mathematics sheer beauty is appealing, the understanding of the world coming from physics simply daunting, but there's something missing. It's at the same time too intangible and too satisfying.

As we sit through the lecture, our phones play a notification sound. It's a new post on the university's Facebook page, something about a startup doing Machine Learning describing some idea for a project. We have vague notions about what Machine Learning is, we're curious, and we also need to find a subject for our capstone project. It's a great opportunity, so we hit reply, without knowing how far this would lead us.

Fast-forward a few years, and I'm working as a ML Engineer for a San Francisco based company, after a detour at Stanford University, something far beyond the imagination of my younger self.

Why am I telling this story?

Because it's commonplace. Lots of my peers have a similar trajectory : a background in some theoretical field (maths, physics, statistics to name a few), all drawn by Machine Learning promises, seeing it as an impactful shortcut from theory to reality, ending up in the field almost accidentally. On paper, it's the perfect job for our profile : a combination of research, science and engineering. A sweet middle spot between the ivory tower of research and the stress-inducing production responsibilities of an engineer.


However, there is a number of things I wish I knew before taking on that path, and at the core of this typical trajectory lie a number of misconceptions that are potentially harmful for the young field of Machine Learning as well as for the professional fulfillment of young adults trying to figure out what they want to do in life. As more students graduate and start working for the industry, the requirements evolve, and they are not necessarily the ones one would expect.

In this post, I want to debunk some of the myths around Machine Learning Engineering, and talk about what I think are the core challenges of the job.

> A lot of ML Engineers come into the field with little to no experience in Software Engineering. Worse, Machine Learning perception as a research-first field hides a painful but simple truth : Machine Learning in the industry is mostly about Software Engineering.




## What school teaches

At the university, Machine Learning classes usually focus on the modeling part and the theoretical foundations. First, it covers topics like linear algebra, convex optimization, gradient descent and the chain-rule. Then, there usually are some discussions about standard problems (predicting the price of a house, classifying images, etc.) and the best-performing models. Maybe some foundational papers reading. At the end of the class, students pair up to work on a cool project in order to apply all the things learned during the quarter : find a dataset, read a few papers, implement them, tweak a few things, and after some time spent resolving over-fitting issues, fixing programming errors, and doing hyper-parameter search, get to a point where the project's code achieves some good score on the task's metric. After writing the final report, as expected, we go on with our lives and the code rarely sees the light of a terminal ever again.

Sure, this is all natural, a mandatory step, and the discipline's core. However, in the industry, challenges go far beyond modeling.

Actually, *modeling is not even where the real challenge is*.

## The first few months on the job

The university years are now over and with great excitement, I start my first job, joining a relatively young company, they've been doing some Machine Learning, but nothing is (yet) set in stone, as the team is still trying to figure out best practices (as the rest of the industry, btw).

As a first project, let's say that the tech lead wants to develop a model that is able to recognize medication side-effects. I feel confident, after all this is pretty similar to some class project I've done at school. I already have ideas about which models to try, what problems I'm likely to face.

I ask : *where can I find the dataset?*

### The dataset problem

And here comes the first surprise : there isn't. How is that possible? I knew datasets had to be built at some point, but this is something I've never done. Well, no problem, let's just find or build a dataset right?

From there, there are 2 typical scenarios.

1. After some googling, I end up on some university or conference webpage that provides just the right dataset. Assuming the copyrights are permissive enough, I download a copy, only to discover that even though the task is similar *on paper*, the data is not quite the same as the one the model will have to deal with. Maybe the company works on free text, long notes, oral transcriptions, but this dataset is a simpler one, with shorter, curated sentences. I decide to give it a shot, but the first experiments are deceptive, and performance does not seem to translate well to real-world data. After discussing the issue with the team, it is decided that a better dataset is needed, at least as a way to evaluate the model on some representative test set.

2. It turns out that there is no dataset out there with the right properties, but thankfully, the company has some unannotated data, and I've heard of a nice tool to help with the annotation process (see [brat](https://brat.nlplab.org), [prodigy](https://prodi.gy)). Motivated, I decide to start labeling a few hundred examples. After a few hours of reading and labeling, I realize that the definition of the label is rather vague : what is a medication side effect ? Should we have one or more labels ? What part of sentences need annotations ? Should prepositions be included ? What about entities that span different parts of a sentence, or even different parts of a paragraph ?

### The first deployment

I've somehow managed to get my hands on a "good dataset" (we'll talk about what this means further on). I start experimenting, and soon I have a decent model. The team seems to think that performance is good enough for a demo, and asks me to "deploy it".

I quickly browse through the code base, and realize there isn't really a standard way to deploy a model. I still manage to find an example, and follow those steps

1. First, I clean the model class, and add it to one of the company's ML packages. I learn about the review process, realize that I need to be more rigorous in my coding style : the pull request won't go through if the linter ([pylint](https://www.pylint.org)) and static type checker ([mypy](http://mypy-lang.org)) do not pass. After a few back-and-forth with the dev lead, the code is finally in a state where it's acceptable to merge. My model class looks like this

    ```python
    class SideEffectModel:

        def __init__(self, side_effects_vocab, tokenizer, weights):
            self.side_effects_vocab = side_effects_vocab
            self.tokenizer = tokenizer
            self.weights = weights

        def tokenize(self, text: str) -> List[str]:
            self.tokenizer.tokenize(text)

        def featurizer1(self, words: List[str]) -> List[Dict]:
            # First featurizer

        def featurizer2(self, words: List[str]) -> List[Dict]:
            # Second featurizer

        def predict(self, text: str) -> List[Tuple[int]]:
            sentences = self.tokenize(text)
            for sentence in sentences:
                features1 = self.featurizer1(sentence)
                features2 = self.featurizer2(sentence)
                # etc.

        def train(self, dataset: List[Tuple[str, List[Tuple[int]]]]):
            # Do stuff
    ```

2. Now other people can import my class and train their own model! However, the model instance trained on the dataset is still sitting somewhere on my laptop. It depends on a given vocabulary, a tokenizer, has some weights, and these are too big to be kept with the code. What should I do?
Of course, the answer is obvious : let's store the vocabulary, weights and tokenizer files on some shared file system, like `s3`or `hdfs`.

    After some process involving copying, migration, permission settings update, I end up in the right state

    ```
    s3://prod/user/guillaume/projects/side-effects/vocabulary.json
    s3://prod/user/guillaume/projects/side-effects/weights.hdf5
    s3://prod/user/guillaume/projects/side-effects/tokenizer.pkl
    ```

3. Now, the artifacts can be used by the code that runs in production. After some investigation, I understand that production models need to be slightly different, and should be able to consume a special data class. So here I am, writing a small wrapper class

    ```python
    class SideEffectModelProdWrapper:

        PATH_VOCAB = "s3://prod/root/user/guillaume/projects/side-effects/vocabulary.json"
        PATH_WEIGHTS = "s3://prod/root/user/guillaume/projects/side-effects/weights.hdf5"
        PATH_TOKENIZER = "s3://prod/root/user/guillaume/projects/side-effects/tokenizer.pkl"

        def __init__(self):
            vocab = reload(PATH_VOCAB)
            weights = reload(PATH_WEIGHTS)
            tokenizer = reload(PATH_TOKENIZER)
            self._model = SideEffectModel(vocab, tokenizer, weights)

        def infer(self, data: ProdDataFormat):
            preds = self._model.predict(data.text)
            data.predictions = ProdPredictionFormat(preds)
            return data
    ```


Mission accomplished. Now the production system can use the `SideEffectModel`!


## Scaling Challenges
