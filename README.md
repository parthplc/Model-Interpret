# Interpreting model output 
In this repository, we have collected example notebooks for using three library.

* SHAP
* LIME
* transformer-interpret

## SHAP
SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see papers for details and citations).
* Installation 

```
pip install shap
or
conda install -c conda-forge shap
```

* Link : [SHAP](https://github.com/slundberg/shap).
The shap folder contains various types of notebooks and their examples to use. 
* One [issue](https://github.com/slundberg/shap/issues/963) with shap is it provides log odds rather than probability. It is the most used and active library for model interpretability currently. 
* You can find example notebooks for most of the general cases inside shap/ folder. 

* For various API usage refer shap/api_examples


## LIME
This project is about explaining what machine learning classifiers (or models) are doing. At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data) or images, with a package called lime (short for local interpretable model-agnostic explanations). Lime is based on the work presented in this [paper](https://arxiv.org/abs/1602.04938) (bibtex here for citation).

<b>Installation</b>

```
pip install lime
```
* Link : [LIME](https://github.com/marcotcr/lime)

* API reference : [Link](https://lime-ml.readthedocs.io/en/latest/)

* Notebooks : lime/

There might be dependency issue while running lime in few example notebooks(especially with keras and huggingface transformers)

## Transformer-Interpret
Transformers Interpret is a model explainability tool designed to work exclusively with the 🤗 transformers package.

In line with the philosophy of the transformers package Tranformers Interpret allows any transformers model to be explained in just two lines. It even supports visualizations in both notebooks and as savable html files.

<b>Installation</b>

```
pip install transformers-interpret
```
Supported:
```
Python >= 3.6
Pytorch >= 1.5.0
transformers >= v3.0.0
captum >= 0.3.1
The package does not work with Python 2.7 or below.
```
* Link : [Transformer-Interpret](https://github.com/cdpierse/transformers-interpret)
