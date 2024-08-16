# Machine Learning Engineer assignment

This codebase contains code for an assignment where it is the goal to predict future sales based on the
data of a loyalty program.

## Dataset analysis
Before I started implementing any feature engineering or predicton algorithms I wanted to get some
understanding of the data. Therefore I wrote a class that does some basic data visualisation: `DataVisualizer`
This class contains several methods that plot certain parts of the data. From these plots I drew the
following conclusions:
* There doesn't seem to be a seasonal trend in the data; the sales revenue and quantity are spread evenly
* Not all users make the same amount of purchases. The amount of purchases seems spread evenly.
* The spread between the sales of a user could be a useful feature, as it is not constant.

## General approach and constructing features
The basic plan is to prepare the data for a classification setup where we want a model that can
output a binary prediction on whether or not a user will make a sale next week. To do this
we need to construct features and targets we can train a classifier with. I chose the following
approach:
 * Split off the last week from the dataset to use to construct training targets
 * Create a DataFrame for all users
 * Derive features for each user and merge them into one row for each user
 * Make binary targets based on whether or not a user made a purchase in the last week
 * Train a model that takes the user features as input and the purchase targets as output
 * Recreate the features for all users but now with the last week included
 * Make a prediction for each user with our trained model

### Constructed features
For each user we want to create a feature vector based on the provided data. I chose the
following features:
 * The date at which the last purchase was made
 * The number of days between the last purchase and the one before that
 * Same feature, but then the diff between the previous purchase and second previous purchase
 * Same feature, but then the diff between the second previous purchase and third previous purchase
 * The average number of days between the last four purchases
 * The standard deviation between the last four purchases

Since not all users made four purchases, some of those fields would be NaN. I filled them with
the value 99 because I reasoned that if a user doesn't make a lot of purchases that user is
less likely to do so in the future.

### Tested models
To keep it simple I tested three different models:
 * Logistic regression
 * Naive Bayes Classifier
 * Random Forest

I trained all of these models on a slice of the data of 80% and then ran three folds of
K-Fold cross valitation. The Random forest scored the best so I picked that one. To get
an estimation on how well my predictions would do I validated the performance of this
model on the held-out 20%. The model got an accuracy of 87.0%, which made me confident
enough to continue. This trained model is what I used for the submitted predictions.

## Production recommendations
To release this to production, several steps need to be taken. Firstly, we need to identify
how the end user would like to interact with this service. A simple solution could be an
API server that hosts this model and has an HTTP endpoint to which customer data could be
sent. Of course, other options are possible.

The important part is how to keep the model up-to-date. Each week new data would become
available and the trained model would need to be adjusted. The derived features could
be updated in a simple way. Important to note that we would use the newly added week
to update our training targets and use the old target week to update the features.
If a customer made a new purchase, all past purchase day diffs could be shifted one
column and the new purchase can be added. If no purchase was made, then no action would
need to be taken.

The currect Random Forest algorithm would need to be fitted/trained from scratch. At the
current scale of number of customers, that is not a problem for a job that runs once per
weeek. If the number of customers would scale by an order of magnitude, we would most
likely be better off with a model that is easier to fine-tune from a pre-trained state.
A Multi-Layer Perceptron would be a good candidate for that.


#### Installation Notes
* Install TKinter correctly if you want the plots
* Install poetry
* Then run `poetry install`
