---
layout: post
title: "Tools and techniques for configurable python code"
description: "Tools and techniques for configuration in python for Machine Learning"
excerpt: "A Machine Learning motivated odyssey"
mathjax: true
comments: true
tags: config python
github: https://github.com/guillaumegenthial/config-python
published: True
---

As a Deep Learning Engineer, I've recently been thinking about clean ways to organize code and define different pipelines. Here is an attempt to summarize my learnings.

<!-- MarkdownTOC -->

* [Introduction](#introduction)
* [A simple use case](#a-simple-use-case)
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


<a id="a-simple-use-case"></a>
## A simple use case

Let's take a simple example : we want to compute `2 * x + 1`.

If I want to quickly put something together, I can just write a short script that does the job

```python
def f(x):
    return 2 * x + 1

f(2)
```

Of course this is a dummy example, but you can extrapolate, increasing complexity. Once your initial script becomes too long, a natural thing to do is to create modules and helper functions, in an attempt to improve code reuse, effectively performing [semantic compression](https://caseymuratori.com/blog_0015). Now you have a script disguised as a "library". And this is perfectly fine, if it's the first iteration of a project, or if you only need to support one use-case.

<a id="a-need-for-configurable-code"></a>
## A need for configurable code

One day, the project manager comes in, asking you to support a new use case : `2 * (2 * x + 1)`.

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

In real life, this means passing combinations of arguments to helper functions, quickly becoming complicated to maintain. As different use cases keep coming, the number of `if ... else ...` statements increases, reaching an unhealthy ratio. Soon, the combination of options forces parts of your code to support a combinatorial number of logic blocks. If you have 2 main options with 10 possibilities each, and each combination of them require some custom code, that's 10 x 10 possibilities! Chances are that in parts of the code that you may be less familiar with, a specific combination of options causes a failure. Hopefully you follow the guidelines of test-driven development and such a liability will be exposed before any release.

<a id="modularizing-your-code"></a>
## Modularizing your code

After a while, the expectations become more generic and you're required to support "all combinations of `2 * x` and `x + 1` operations". Worse, you see a near future where other integer operations will need to be supported, like `3 * x`, and that you probably need to isolate each of these transforms as well as how you combine them together.

After some time spent rewriting parts of the code to make it more modular, you come to the conclusion that each of these transforms is independent from the others and that combining them together is another issue ([Separation of Concern](https://en.wikipedia.org/wiki/Separation_of_concerns)). Each transform should be responsible of one thing and one thing only ([Single Responsibility Principle](https://en.wikipedia.org/wiki/Single_responsibility_principle)), while following the same contract.

In python, one the right ways of doing this is to define an "interface" for your transforms, using an abstract class

```python
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    @abstractmethod
    def apply(self, x):
        raise NotImplementedError()
```

and implement different versions of that base class

```python
class PlusOneTransform(BaseTransform):
    def apply(self, x):
        return x + 1

class TimesTwoTransform(BaseTransform):
    def apply(self, x):
        return 2 * x
```

Finally, chaining the transforms is the job of some other class (which itself is a transform)

```python
class Chain(BaseTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, x):
        for transform in self.transforms:
            x = transform.apply(x)
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
times_two = TimesTwoTransform()
plus_one = PlusOneTransform()
chain = Chain([times_two, plus_one])
chain.apply(2)
```

and

```python
# use-case-two.py
times_two = TimesTwoTransform()
plus_one = PlusOneTransform()
chain = Chain([times_two, plus_one, times_two])
chain.apply(2)
```

At this point, it may look like we haven't made a lot of progress. It turns out that in the process of making our code modular and reusable in a nice an abstract way
- we created a library (a collection of tools that can be easily re-used outside), in other words, we created a high-level API that users can use ([How to design a good API and why it matters](https://www.youtube.com/watch?v=aAb7hSCtvGw))
- we separated the transforms' __implementation__ (don't forget that this is a dummy example but the actual transforms you are implementing are much more complicated) from the __usage__. Actually, each of these scripts can be seen as a special configuration.

> For example, the workflow manager [airflow](https://airflow.apache.org/) lets you define pipelines using a python interface (Directed Acyclic Graphs, Operators, etc.). The documentation says : *One thing to wrap your head around (it may not be very intuitive for everyone at first) is that [Airflow Python scripts are] really just configuration files specifying DAG’s structure as code*

While this may sound obvious, it's crucial to separate implementation from usage, especially because it's so easy to mix the two, and end up with a library that is part script-like and usage-specific, side-by-side with a collection of helper functions that may have otherwise been reusable for a wider variety of use-cases.

<a id="with-config-files"></a>
### With config files

While python files are probably sufficient in most cases (and this should probably always be possible because pipeline creators are likely to be programmers like you), some situations might benefit from the use of a more convenient pipeline definition format. Advantages and requirements may include
- avoid duplication by splitting configs into sub-configs.
- use a format that can easily be shared
- provide a way for non-programmers to define their own pipelines
- provide a lightweight, less-verbose way of defining pipeline

There are a number of good formats that are widely adopted in the python community
- `.json` (JavaScript Object Notation), probably the most popular format, as it naturally resembles python dictionaries.
- [`.jsonnet`](https://jsonnet.org), built on top of json, adds support for imports, variable definition and much more, before "compilation" to a standard `.json`.
- `.ini` (used by [configparser](https://docs.python.org/3/library/configparser.html))
- `.yaml`
- `.xml`

Having said that, the question becomes : what do we write in these configuration files, and how do we reload them?

Usually, the first step would be to implement a way to translate a python dictionary into an object. There are multiple ways of doing it
- define a `DictSerializable` interface and have each class implement a class method `from_dict(cls, data)` that creates an object from a dictionary.
```python
class DictSerializable(ABC):

    @abstractclassmethod
    def from_dict(cls, data):
        raise NotImplementedError()
```
For example, for the `Chain` class it might look like
```python
class Chain(BaseTransform, DictSerializable):

    @classmethod
    def from_dict(cls, data):
        transforms = []
        for transform_name in data["transforms"]:
            if transform_name == "times_two":
                transforms.append(TimesTwoTransform())
            elif transform_name == "plus_one":
                transforms.append(PlusOneTransform())
            else:
                raise ValueError()
        return Chain(transforms)
```
> This is basically what the `FromParams` class does in the [AllenNLP](https://github.com/allenai/allennlp) library.

- use a `Schema` approach. In other words, delegate the creation of objects from dictionaries to another class. This is maybe the most widely-used approach, but might be overkill in some cases. Have a look at the [marshmallow](https://marshmallow.readthedocs.io/en/stable/) library for example.

Another tip : you might want to validate and normalize the dictionaries before creating instances from them. This can be useful to check for missing entries, fill-out default values etc. I've been using [cerberus](https://docs.python-cerberus.org/en/stable/) for that purpose.

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

Sharing and editing different pipelines is now even easier! In a way, our json syntax is some kind of small "programming language" that lets us interface with our library in a minimalistic way.

<a id="a-more-complicated-example"></a>
## A more complicated example

In the previous example, things were simple. We had very few classes, with reasonable dependencies. There was not a lot of parameters, object nesting, etc.

Let's take a slightly more complicated example.

Let's require each `Transform` to define a `name` attribute (this illustrates that dependencies usually have their own parameters), as well as depend on a `Vocab` instance that will be shared among the transforms (this illustrates the need to support arbitrary hierarchies of dependencies).

In other words, we modify the code in the following way

```python
class Vocab:
    def __init__(self, words):
        self.words


class BaseTransform(ABC):
    def __init__(self, name, vocab):
        self.name = name
        self.vocab = vocab
```

Now, defining our pipeline in python is still straightforward (and that's why the first step towards configuration is to use plain python)

```python
vocab = Vocab(["foo", "bar"])
times_two = TimesTwoTransform("times_two", vocab)
plus_one = PlusOneTransform("plus_one", vocab)
chain = Chain([times_two, plus_one])
chain.apply(2)
```

But what about our nice `json` format? If we adopt a backwards engineering approach, we can sketch what it could look like.

```json
{
    "transforms": [
        {
            "type": "TimesTwoTransform",
            "params": {
                "name": "times_two",
                "vocab": ["foo", "bar"]
            }
        },
        {
            "type": "PlusOneTransform",
            "params": {
                "name": "plus_one",
                "vocab": ["foo", "bar"]
            }
        }
    ]
}
```

The config file now contains almost all the necessary information. We can infer the `Transform` classes using the `"type"` entry, and use the `"params"` to create instances of those classes. Let's do it for the sake of completeness


```python
class BaseTransform(DictSerializable):

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], Vocab(data["vocab"]))


class Chain(BaseTransform, DictSerializable):

    @classmethod
    def from_dict(cls, data):
        transforms = []
        for d in data["transforms"]:
            if d["type"] == "TimesTwoTransform":
                transforms.append(TimesTwoTransform.from_dict(d["params"]))
            elif d["type"] == "PlusOneTransform":
                transforms.append(PlusOneTransform.from_dict(d["params"]))
            else:
                raise ValueError()
        return Chain(transforms)
```

> There are ways to improve the whole logic, for example we might use inspection to directly resolve the class from its name or full import string, or make the `Vocab` class also `DictSerializable`.

It seems that we have achieved our goal, haven't we?

Actually, there is an issue with the way the vocabulary is created : we actually created two identical yet distinct instances of the same vocabulary, while what we want is to share the same object between the transforms (in Natural Language Processing for instance, components of a library may want to use the same vocabulary, and more generally some dependencies might be resource intensive and you want to avoid wasting resources).

This is almost a singleton kind of situation (almost, because we might have other vocabs elsewhere, it just turns out that these transforms need to share this one instance) and we can expect this kind of dependency to come up in different places.

> We could modify our json schema to capture this information, update our `from_dict` method, add some convention for object's reuse and singletons, etc.

<a id="configuration-as-dependency-injection"></a>
### Configuration as Dependency Injection

This whole configuration process is actually a [Dependency Injection](https://en.wikipedia.org/wiki/Dependency_injection) problem.

#### What Dependency Injection mean

We want to create a pipeline made of components that hierarchically depend on each other. We want a way to create dependencies and inject them when creating objects that depend on it.

In our example, first we need to create the `Vocab`, then create the different `Transforms` and "inject" the vocab dependency at creation time, and finally provide the transforms when creating the `Chain` pipeline.

There are multiple ways of effectively implementing dependency injection. Our `from_dict` approach, though imperfect, could be improved to a state where it supports singletons, scoping etc.

#### Dependency Injection using Registries and Assemblers

In our example, the complexity stems from the multiple dependencies, and the fact that some objects are shared (the `Vocab` is the same for all our `Transform`).

A way to deal with a complex dependency pattern is to change the code and delegate the injection to specialized classes. For example, in the `Vocab` case, we can create a `VocabRegistry` in charge of providing the objects by name.


```python
class VocabRegistry:

    VOCABS = dict()

    @staticmethod
    def get(name):
        return VOCABS[name]
```

and update the `BaseTransform` into

```python
class BaseTransform(ABC, DictSerializable):

    def __init__(self, name, vocab_name):
        self.name = name
        self.vocab = VocabRegistry.get(vocab_name)
```

That way, the objects can be created independently, the "injection" being the Registry's responsibility.

Another (close) way of resolving the complex dependencies is to have an assembler that puts everything together

```python
class Assembler:

    @staticmethod
    def chain_from_dict(data):
        # First create the vocab
        vocab = Vocab.from_dict(data["vocab"])

        # Then, create transforms
        transforms = []
        for d in data["transforms"]:
            d["vocab"] = vocab  # The vocab dependency is injected here
            if d["type"] == "TimesTwoTransform":
                transforms.append(TimesTwoTransform.from_dict(d["params"]))
            elif d["type"] == "PlusOneTransform":
                transforms.append(PlusOneTransform.from_dict(d["params"]))
            else:
                raise ValueError()
        return Chain(transforms)
```

One of the issues of this approach is that it requires you to define assemblers or registries for each of your pipeline types. If in a lot of cases, this won't be too much of a problem, it can be limiting and forbid innovative uses of your library.

#### Dependency Injection using gin-config

The above solution requires a rather counter-intuitive redesign of the code, while increasing complexity to the detriment of readability. In some cases, this is fine, especially if intricate dependencies are not that frequent in the code.

In general, we would like to stick to simple abstractions, with dependencies being defined through the constructor (the `__init__` method). This is at the same time more pythonic and easier to read.

It turns out that there exists nice solutions for dependency injection through config files in python. One of the best tools I've seen is [gin-config](https://github.com/google/gin-config), a dependency injection package for python built by Google.

If we were using `gin`, the only thing we need to do is define a `.gin` configuration file, that defines all the dependencies in a nice, lightweight and composable syntax. After having annotated each of our class with a special decorator `@gin.configurable`, that allows you to tweak and define the dependencies (here our example is simple enough so that we don't need to customize the decorator, but in some cases you might want to rename dependencies, provide defaults, require some dependencies to be defined even though the `__init__` method has a default, etc.)

```python
@gin.configurable
class Vocab:
    pass

@gin.configurable
class TimesTwoTransform:
    pass


@gin.configurable
class PlusOneTransform:
    pass


@gin.configurable
class Chain:
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

Gin supports a lot of functionality that you will probably need
- scoping (define dependencies between object types in different scopes).
- import (compose different gin files into one big pipeline).
<a id="--singletons-define-one-object-that-will-be-reused-in-our-example-we-need-it-for-the-shared-vocabulary"></a>
- singletons (define one object that will be reused, in our example we need it for the shared vocabulary).
- Tensorflow and PyTorch specific functionality


Parsing a gin-file in python is straightforward

```python
import gin
gin.parse_config_file("config.gin")
model = Model()  # Argument `layers` provided by gin
```

The only caveat with gin is that it blurs lines between the code and configuration : the gin syntax is really close to python. Also, because gin is taking care of dependencies for you, it is counter-intuitive at first. However, if you take some time to think about it, gin really provides something that python does not : easy definition of dependencies in a linear way, i.e. define each object, and delegate the injection to gin.

In the above example, notice how we do not explicitly provide the vocab to the transforms : we let gin take care of it.


<a id="a-machine-learning-perspective"></a>
## A Machine Learning Perspective

In Machine Learning, more especially in Deep Learning, your code is usually built around the following abstractions
- dataset
- preprocessing
- layers
- models
- optimizers
- initializers
- losses

Defining a model is usually just chaining layers, define a loss, an optimizer, and combine everything into one function call. More importantly, as an empiric field, being able to quickly test different configurations is key.

Interestingly, it seems that the relatively young world of Deep Learning libraries is converging to a common approach.
- the ability to define pipelines in simple `.json` files provided by [AllenNLP](https://github.com/allenai/allennlp) was key to its success. The implementation of this feature is actually very close to the `from_dict` approach covered in the example of this article.
- similarly, the team behind SpaCy, released a great package to help train neural networks for NLP on top of Tensorflow, PyTorch and JAX : [thinc](). It builds on top of `.ini` configuration files, in a way similar to `gin`.
- more recently, as we see a burst of new packages ([TRAX](), [Flax](), [Haiku]()) in the Deep Learning world motivated by the growth in popularity of [JAX]()(NumPy accelerated on GPU, with higher order differentiation, proper control over random generators, XLA support and a few other tricks), I was glad to see that TRAX was actually using `gin` as a configuration language

<a id="conclusion"></a>
## Conclusion

Here are the main takeaways
- Python and Machine Learning are not an excuse to build poorly designed libraries. We should read about [design patterns](), and try to follow the Single Responsibility Principle and Separation of Concerns principles. It will make the code more modular
- Separate implementation from usage, i.e. build libraries that allow users to do complex things with little code.
- Usage is equivalent to configuration, and configuration boils down to dependency injection.
- While simple python scripts are a great way to define configs, you might want to build some custom `.json` (or any other format that suits your needs) interface for ease-of-use, or switch to `gin` (or any other dependency injection package) for out-of-the-box functionality.

