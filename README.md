# fairkit-learn fairness toolkit

Fairkit-learn is an open-source, publicly available Python toolkit designed
to help data scientists evaluate and explore machine learning models with
respect to quality and fairness metrics simultaneously.

Fairkit-learn builds on top of [scikit-learn](https://scikit-learn.org/stable/), the state-of-the-art tool suite
for data mining and data analysis, and [AI Fairness 360](https://aif360.mybluemix.net/), the state-of-the-art
Python toolkit for examining, reporting, and mitigating machine learning bias
in individual models. 

Fairkit-learn supports all metrics and learning algorithms available in scikit-learn and AI Fairness
360, and all of the bias mitigating pre- and post-processing algorithms available in AI Fairness 360, and provides extension points to add more metrics and algorithms.

# Installation

To install fairkit-learn, run the following command:

``` pip install fairkit_learn==1.8```

# Using fairkit-learn

To use fairkit-learn, first run the following command to install necessary pacakges:

```pip install -r requirements.txt```

Sample code for how to use fairkit-learn can be found in the examples
folder (e.g., Fairkit_learn_Tutorial) in the repo.
