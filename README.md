# Interpreting model output 
In this repository, we have collected example notebooks for using three library.

* SHAP
* LIME
* transformer-interpret

There are two types of model interpretation provided by below repository. 
* Global -> Interpretation of Global impact of individual feaure/columns on complete dataset at once.(Gives something like feature importance)
* Local -> Intrepret the impact of each feature/column on a single row (Importance of feature in predicting that single row.)

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
* Type of Interpretation : Global and Individual row wise interpretation.
* For various API usage refer shap/api_examples
* Expected input : Model Object(linear,treebased,transformer any) and Dataset.
```
explainer = shap.TreeExplainer(model,data=train_data,model_output="probability")
# model_output could be probability,raw,logloss
```
* Output : Shapvalues,log loss or probability per feature proportional to their contribution. [Link](https://github.com/slundberg/shap/blob/master/notebooks/api_examples/explainers/Exact.ipynb)



* Only for tree based model we can have find probability of each feature in final contribution.Here is the [notebook](https://github.com/parthplc/Model-Interpret/blob/master/shap/tabular_examples/tree_based_models/LightGBM%20SHAP%20probability.ipynb).



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
* Model Interpretation : Only row wise interpretation possible
* Input : feature_names,target_columns, catgeorical_feature_names and trained model object
```
exp = explainer.explain_instance(test[i], rf.predict_proba)
# test[i] : ith test example,rf = actual trained model
```
* Output :  return contribution of each feature to final target values.
```
Output various plots to explain the prediction of that given row.
```
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
* Link : [Transformer-Interpret](https://github.com/cdpierse/transformers-interpret)\
* Model Interpretation : Local only

* Input : model,tokenizer for transformer model
```
cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)
# model = Any transfomer model object and tokenizer = Model tokenizer used.
```

* Output : Word wise contribution for the final target class
```
returns value of impact each word in text has on output.
```

