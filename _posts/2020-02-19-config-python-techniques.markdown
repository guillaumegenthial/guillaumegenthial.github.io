---
layout: post
title: "Techniques for configurable python code"
description: "Techniques for configuration in python, a discussion about Dependency Injection motivated by Machine Learning"
excerpt: "A Machine Learning motivated odyssey"
mathjax: true
comments: true
tags: config python
github: https://github.com/guillaumegenthial/config-python
published: True
---

<div class="alert alert-success" role="alert">
  <p>
      This post goes down a dangerous path. Since writing it, I've experienced more real-life situations and my perspective is different.
  </p>
  <p>
    I now believe that the best way to write configurable code is <strong>not to use config files</strong>, but to use code. Every config language ends up being just a broken programming language. It's better in almost every way to use a battle-tested language rather than a DSL that nobody knows. Django does it ; AWS does it with CDK (IaC in TypeScript); etc.
  </p>
  <p>
    See this interesting post : <a href="https://mikehadlow.blogspot.com/2012/05/configuration-complexity-clock.html?m=1">The configuration complexity clock</a>
  </p>
  <hr>
  <p class="mb-0">
    TL;DR : don't do anything fancy, use code. Or maybe keep it super simple with a flat structure (only key-values)...
  </p>
</div>

As a Deep Learning Engineer, I've recently been thinking about clean ways to organize code and define pipelines. Here is an attempt to summarize my learnings.

<!-- MarkdownTOC -->

* [Introduction](#introduction)
* [A simple example](#a-simple-example)
* [A need for configurable code](#a-need-for-configurable-code)
* [Modularizing your code](#modularizing-your-code)
* [Using the modularized code](#using-the-modularized-code)
    - [With python scripts outside the library](#with-python-scripts-outside-the-library)
    - [With config files](#with-config-files)
* [A more complicated example](#a-more-complicated-example)
    - [Configuration as Dependency Injection](#configuration-as-dependency-injection)
* [A Machine Learning Perspective](#a-machine-learning-perspective)
* [Conclusion](#conclusion)

<!-- /MarkdownTOC -->


<a id="introduction"></a>
## Introduction

Let's start with the facts : writing code for a side-project or a class-project is very different from writing code for a large organization. Among other things, you face the following requirements :
- __collaboration__ : it's not only your code, so you need to make sure that the components you add play nicely with your teammates'
- __compatibility over time__ : not only does your new feature need to work, it also has to integrate nicely with the existing logic, without breaking everything. What's more, in the future, you want to avoid having to modify and update your code.
- __flexibility in usage__ : you need to keep in mind that specifications are likely to evolve in the future, and you need to plan accordingly (as much as possible without overdoing it), as you might need to support a wider variety of use cases. Ideally, the design itself should be modular enough so that adding new functionality is easy.


While the previous points may seem obvious, it requires patience and experience to successfully address them, especially if you write python code : as a scripting language by nature, it's easy to forget good software-engineering practices and dive right in. After all, running code is better than nice-but-unusable code.


<a id="a-simple-example"></a>
## A simple example

Let's take a simple example : we want to compute `2 * x + 1`.

If I want to quickly put something together, I can just write a short script that does the job

```python
def f(x):
    return 2 * x + 1

f(2)
```

Of course this is a dummy example, but you can extrapolate, until you reach the complexity of an actual program. Once your initial script becomes too long, a natural thing to do is to create modules and helper functions, in an attempt to improve code reuse, effectively performing [semantic compression](https://caseymuratori.com/blog_0015). Now you have a script disguised as a "library". And this is perfectly fine, if it's the first iteration of a project, or if you only need to support one use case.

<a id="a-need-for-configurable-code"></a>
## A need for configurable code

One day, the project manager comes to see you, asking to support a new use case : `2 * (2 * x + 1)`.

That's easy, let's do something like

```python
def f(x, use_case):
    if use_case == "use_case_1":
        return 2 * x + 1
    elif use_case == "use_case_2":
        return 2 * (2 * x + 1)
    else:
        raise ValueError("Use case not supported.")

f(2, "use_case_2")
```

In real life, this means passing combinations of arguments to helper functions, resulting in something quite complicated to maintain. As different use cases keep coming, the number of `if ... else ...` statements increases, reaching an unhealthy ratio. Soon, the combination of options forces parts of your code to support a combinatorial number of possibilities. If you have 2 main options with 10 possibilities each, and each combination requires some custom logic, that's 10 x 10 possibilities! Chances are that in parts of the code that you may be less familiar with, a specific combination of options causes a failure. Hopefully you follow the guidelines of test-driven development and such a liability will be exposed before any release.

<a id="modularizing-your-code"></a>
## Modularizing your code

After a while, the expectations become more generic and you're required to support "all combinations of `2 * x` and `x + 1`". Worse, you see a near future where other operations will have to be supported, like `3 * x`. You should probably isolate each of these operations from each other as well as how you combine them together.

After some time spent rewriting parts of the code to make it more modular, you come to the conclusion that each of these operations is independent from the others and that combining them together is another issue ([Separation of Concern](https://en.wikipedia.org/wiki/Separation_of_concerns)). Each operation should be responsible of one thing and one thing only ([Single Responsibility Principle](https://en.wikipedia.org/wiki/Single_responsibility_principle)), while following the same contract.

In python, one the right ways of doing this is to define an "interface" for your operations (let's call them `layers`), using an abstract class

```python
from abc import ABC, abstractmethod


class BaseLayer(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
```

and implement different versions of that base class

```python
class PlusOneLayer(BaseLayer):

    def forward(self, x):
        return x + 1

class TimesTwoLayer(BaseLayer):

    def forward(self, x):
        return 2 * x
```

Finally, chaining the layers is the job of some other class

```python
class Model:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

In this example we used object-oriented programming to modularize the code, but other options are also possible (using functions with similar signatures for example). However, the nice thing about OOP is that it lets you define contracts. Once implemented, your editor will help you find errors and it might speedup the whole development process while improving code robustness.

<a id="using-the-modularized-code"></a>
## Using the modularized code

Okay so now we have abstracted and isolated the different components of the program. How do we use it?

<a id="with-python-scripts-outside-the-library"></a>
### With python scripts outside the library

The easiest way is to let the different users of your library define one script per pipeline. For each of the previous use cases, we end up with a script

```python
# use-case-one.py
times_two = TimesTwoLayer()
plus_one = PlusOneLayer()
model = Model([times_two, plus_one])
model.forward(2)
```

and

```python
# use-case-two.py
times_two = TimesTwoLayer()
plus_one = PlusOneLayer()
model = Model([times_two, plus_one, times_two])
model.forward(2)
```

At this point, it may look like we haven't made a lot of progress. It turns out that in the process of making our code modular and reusable in a nice an abstract way
- we created a library (a collection of tools that can be easily re-used), in other words, we created a high-level API that users can build on (see [how to design a good API and why it matters](https://www.youtube.com/watch?v=aAb7hSCtvGw))
- we separated the layers' __implementation__ (don't forget that this is a dummy example but the actual operations you are implementing are much more complicated) from __usage__. Actually, each of these scripts can be seen as a special configuration.

> It is interesting to notice that the workflow manager [airflow](https://airflow.apache.org/) lets you define pipelines using a python interface (Directed Acyclic Graphs, Operators, etc.). From the documentation : *"One thing to wrap your head around (it may not be very intuitive for everyone at first) is that [Airflow Python scripts are] really just configuration files specifying DAG’s structure as code"*

While this may sound obvious, it is crucial to separate implementation from usage, especially because it's so easy to mix the two and end up with a library that is part script-like and usage-specific, side-by-side with a collection of helper functions that may have otherwise been reusable for a wider variety of use cases.

<a id="with-config-files"></a>
### With config files

While python files are probably sufficient in most cases (and this should probably always be possible because pipeline creators are likely to be programmers, and who knows what they will have in mind), some situations might benefit from the use of a more convenient pipeline definition format. Advantages and requirements may include
- Avoid duplication by splitting configs into sub-configs.
- Use a format that can easily be shared.
- Provide a way for non-programmers to define their own pipelines.
- Provide a lightweight, less-verbose way of defining pipelines.

There are a number of good formats that are widely adopted in the python community
- `.json` (JavaScript Object Notation), probably the most popular format, as it resembles python dictionaries.
- [`.jsonnet`](https://jsonnet.org), built on top of json, adds support for imports, variable definition and much more, before "compilation" to a standard `.json`.
- `.ini` (see [configparser](https://docs.python.org/3/library/configparser.html))
- `.yaml`
- `.xml`

Having said that, the question becomes : what do we write in these configuration files, and how do we reload them?

Usually, the first step would be to implement a way to create an object from a python dictionary. There are multiple ways of doing it
- define a `Serializable` interface and have each class implement a class method `from_params(cls, params)` that creates an object from a dictionary.
```python
class Serializable(ABC):

    @abstractclassmethod
    def from_params(cls, params):
        raise NotImplementedError()
```
For example
```python
class Model(Serializable):

    @classmethod
    def from_params(cls, params):
        transforms = []
        for transform_name in params["transforms"]:
            if transform_name == "times_two":
                transforms.append(TimesTwoLayer())
            elif transform_name == "plus_one":
                transforms.append(PlusOneLayer())
            else:
                raise ValueError()
        return Model(transforms)
```
> This is basically what the `FromParams` class does in the [AllenNLP](https://github.com/allenai/allennlp) library.

- use a `Schema` approach. In other words, delegate the creation of objects from dictionaries to another class. This is probably the most common approach, but might be overkill in some cases. Have a look at the great [marshmallow](https://marshmallow.readthedocs.io/en/stable/) library.

Another tip : you might want to validate and normalize the dictionaries before creating instances from them. This can be useful to check for missing entries, fill-out default values, rename parameters, etc. I've been using [cerberus](https://docs.python-cerberus.org/en/stable/) for that purpose.

Now, our different use cases can be defined in simple `.json` files

```json
{
    "transforms": ["times_two", "plus_one"]
}
```

and

```json
{
    "transforms": ["times_two", "plus_one", "times_two"]
}
```

Sharing and editing different pipelines is now even easier! In a way, our `json` syntax is some kind of small "programming language" that lets us interface with our library in a minimalistic and convenient way.

<a id="a-more-complicated-example"></a>
## A more complicated example

In the previous example, things were simple. We had very few classes, with reasonable dependencies. There was not a lot of parameters, object nesting, etc.

Let's take a slightly more complicated example.

Let's require each `Layer` to define a `name` attribute (this illustrates that dependencies usually have their own parameters), as well as depend on a `Vocab` instance that will be shared among layers (this illustrates the need to support arbitrary hierarchies of dependencies).

In other words, we modify the code in the following way

```python
class Vocab:
    def __init__(self, words):
        self.words


class BaseLayer(ABC):
    def __init__(self, name, vocab):
        self.name = name
        self.vocab = vocab
```
> Technically, in this example, the layers won't use the vocab as it only represents some dependency they might have. In NLP, having a single vocabulary used in various places is very common, hence this example.

Now, defining our pipeline in python is still straightforward (and that's why the first step towards configuration is to use plain python)

```python
vocab = Vocab(["foo", "bar"])
times_two = TimesTwoLayer("times_two", vocab)
plus_one = PlusOneLayer("plus_one", vocab)
model = Model([times_two, plus_one])
model.forward(2)
```

But what about our nice `json` format? If we adopt a backwards engineering approach, we can sketch what it could look like.

```json
{
    "transforms": [
        {
            "type": "TimesTwoLayer",
            "params": {
                "name": "times_two",
                "vocab": ["foo", "bar"]
            }
        },
        {
            "type": "PlusOneLayer",
            "params": {
                "name": "plus_one",
                "vocab": ["foo", "bar"]
            }
        }
    ]
}
```

The config file now contains almost all the necessary information. We can infer the `Layer` classes using the `"type"` entry, and use the `"params"` to create instances of those classes. Let's do it for the sake of completeness


```python
class BaseLayer(Serializable):

    @classmethod
    def from_params(cls, params):
        return cls(params["name"], Vocab(params["vocab"]))


class Model(BaseLayer, Serializable):

    @classmethod
    def from_params(cls, params):
        transforms = []
        for d in params["transforms"]:
            if d["type"] == "TimesTwoLayer":
                transforms.append(TimesTwoLayer.from_params(d["params"]))
            elif d["type"] == "PlusOneLayer":
                transforms.append(PlusOneLayer.from_params(d["params"]))
            else:
                raise ValueError()
        return Model(transforms)
```

> There are ways to improve the whole logic, for example we might use inspection to automatically resolve the class from its name or import string, or make the `Vocab` class also `Serializable`.

It seems that we have achieved our goal, haven't we?

Actually, there is an issue with the way the vocabulary is created : we actually created two identical yet distinct instances of the same vocabulary, while what we want is to share the same object between the transforms (generally some dependencies might be resource intensive and you want to avoid wasting resources).

This is almost a singleton kind of situation (almost, because we might have other vocabs elsewhere, it just turns out that these transforms need to share this one instance) and we can expect this kind of dependency to come up in different places.

> We could modify our json schema to capture this information, update our `from_params` method, add some convention for object's reuse and singletons, etc., but this goes beyond the scope of this post.

<a id="configuration-as-dependency-injection"></a>
### Configuration as Dependency Injection

This whole configuration process is actually a [Dependency Injection](https://en.wikipedia.org/wiki/Dependency_injection) problem.

#### What Dependency Injection mean

We want to create a pipeline made of components that hierarchically depend on each other. We want a way to create dependencies and inject them when creating objects that depend on it.

In our example, first we need to create the `Vocab`, then create the different `Layers`, "inject" the vocab dependency at creation time, and finally provide the layers when creating the `Model`.

There are multiple ways of effectively implementing dependency injection. Our `from_params` approach, though imperfect, could be improved to a state where it supports singletons, scoping etc.

#### Dependency Injection using Registries and Assemblers

In our example, the complexity stems from the multiple dependencies, and the fact that some objects are shared (the `Vocab` is the same for all our `Layer`).

A way to deal with a complex dependency pattern is to change the code and delegate the injection to specialized classes. For example, in the `Vocab` case, we can create a `VocabRegistry` in charge of providing the objects by name.


```python
class VocabRegistry:

    VOCABS = dict()

    @staticmethod
    def get(name):
        return VOCABS[name]
```

and update the `BaseLayer` into

```python
class BaseLayer(ABC, Serializable):

    def __init__(self, name, vocab_name):
        self.name = name
        self.vocab = VocabRegistry.get(vocab_name)
```

That way, the objects can be created independently, the "injection" being the Registry's responsibility.

Another (close) way of resolving the complex dependencies is to have an assembler that puts everything together

```python
class ModelAssembler:

    @staticmethod
    def from_dict(data: Dict) -> Model:
        # Create vocab
        vocab = Vocab(data["vocab"])

        # Create layers
        layers = []
        for layer_data in data["layers"]:
            layer_type = layer_data["layer_type"]
            layer_name = layer_data["layer_name"]
            if layer_type == "TimesTwoLayer":
                layer = TimesTwoLayer(layer_name, vocab)
            elif layer_type == "PlusOneLayer":
                layer = PlusOneLayer(layer_name, vocab)
            else:
                raise ValueError(f"{layer_type} not recognized.")
            layers.append(layer)
        return Model(layers)
```

One of the issues of this approach is that it requires you to define assemblers or registries for each of your pipeline dependencies. If in a lot of cases, this won't be too much of a problem, it can be limiting and forbid innovative uses of your library.

#### Dependency Injection using gin-config

The above solution requires a rather counter-intuitive redesign of the code, while increasing complexity to the detriment of readability. In some cases, this is fine, especially if intricate dependencies are not that frequent in the code.

In general, we would like to stick to simple abstractions, with dependencies being injected through the constructor (the `__init__` method). This is at the same time more pythonic and easier to read.

It turns out that there exists nice solutions for dependency injection through config files in python. One of the best tools I've seen is [gin-config](https://github.com/google/gin-config), a dependency injection package for python built by Google.

If we were using `gin`, the only thing we need to do is define a `.gin` configuration file, that sets all the dependencies in a nice, lightweight, and composable syntax. After having annotated all the classes with a special decorator `@gin.configurable` that allows you to tweak and define the dependencies (here our example is simple enough so that we don't need to customize the decorator, but in some cases you might want to rename dependencies, provide defaults, require some dependencies to be defined even though the `__init__` method has a default, etc.)

```python
@gin.configurable
class Vocab:
    pass


@gin.configurable
class TimesTwoLayer:
    pass


@gin.configurable
class PlusOneLayer:
    pass


@gin.configurable
class Model:
    pass
```

Here is what the `.gin` config file would look like in our case


```python
import libnn.layers
import libnn.model
import libnn.vocab

# Vocab Singleton
# =====================================================================
Vocab.words = ["foo", "bar"]
vocab/singleton.constructor = @Vocab

# Layers Scopes
# =====================================================================
layer1/TimesTwoLayer.name = "times_two"
layer1/TimesTwoLayer.vocab = @vocab/singleton()
layer2/PlusOneLayer.name = "plus_one"
layer2/PlusOneLayer.vocab = @vocab/singleton()


# Model
# =====================================================================
Model.layers = [@layer1/TimesTwoLayer(), @layer2/PlusOneLayer()]
```
> Gin's philosophy is a little confusing at first, as it defines dependencies at the class level. If you need different objects of the same class, you need to scope the dependencies.

Gin supports a lot of functionality that you will probably need
- scoping (define dependencies between object types in different scopes).
- import (compose different gin files into one big pipeline).
- singletons (define one object that will be reused, in our example we need it for the shared vocabulary).
- Tensorflow and PyTorch specific functionality


Parsing a gin-file in python is straightforward

```python
import gin
gin.parse_config_file("config.gin")
model = Model()  # Argument `layers` provided by gin
```

The only caveat with gin is that it blurs lines between code and configuration : the gin syntax is really close to python. Also, because gin is taking care of dependencies for you, it is counter-intuitive at first. However, if you take some time to think about it, gin really provides something that python does not : easy definition of dependencies in a linear way, i.e. define how classes are supposed to be put together, and delegate the injection to gin.


#### With python `super()` method

Not much to say here, as everything is explained by Raymon Hettinger in [Super considered super!](https://www.youtube.com/watch?v=EiOglTERPEo) PyCon 2015 talk.


<a id="a-machine-learning-perspective"></a>
## A Machine Learning Perspective

In Machine Learning, more especially in Deep Learning, the code is usually built around the following abstractions
- dataset
- preprocessing
- layers
- models
- optimizers
- initializers
- losses

Defining a model is usually just chaining layers, defining a loss, an optimizer, and combining everything into one function call. More importantly, as an empiric field, being able to quickly test different configurations is key.

Interestingly, it seems that the relatively young world of Deep Learning libraries is converging to a common approach.
- the ability to define pipelines in simple `.jsonnet` files provided by [AllenNLP](https://github.com/allenai/allennlp) was key to its success. The implementation of this feature is actually very close to the `from_params` approach covered in the example of this article.
- similarly, the team behind SpaCy released a great package to help train neural networks for NLP on top of Tensorflow, PyTorch and JAX : [thinc](https://thinc.ai/). It builds on top of `.ini` configuration files, in a similar manner to `gin`.
- more recently, as we see a burst of new packages ([Trax](https://github.com/google/trax), [Flax](https://github.com/google-research/flax), [Haiku](https://github.com/deepmind/dm-haiku)) in the Deep Learning world motivated by the growth in popularity of [JAX]() (NumPy accelerated on GPU, with higher order differentiation, proper control over random generators, XLA support and a few other tricks), I was glad to see that Google Brain's [Trax](https://github.com/google/trax) was actually using `gin` as a configuration language.

<a id="conclusion"></a>
## Conclusion

Here are the main takeaways
- Python and Machine Learning are not an excuse to build poorly designed libraries. We should read about [design patterns](), and try to follow the [Single Responsibility](https://en.wikipedia.org/wiki/Single_responsibility_principle) and [Separation of Concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) principles. It will make the code more modular
- Separate implementation from usage, i.e. build libraries that allow users to do complex things with little code.
- Usage is equivalent to configuration, and configuration boils down to dependency injection.
- While simple python scripts are a great way to define configs, you might want to build some custom `.json` (or any other format that suits your needs) interface for ease-of-use, or switch to `gin` (or any other dependency injection package) for out-of-the-box functionality.

