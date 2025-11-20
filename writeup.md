# First Draft Writeup

## Models Chosen

I chose to initially include as many models as possible, as it makes it much easier to evaluate which ones perform well and which ones don't. The data is structured in a way that makes it easy to add additional models, so I figured the more the merrier :)

Due to that, I tried the following models:

-   KNN
-   Multi-layer Perceptron
-   Naive Bayes
-   Random Forest
-   Support Vector Machine

For a baseline, I included a "simple" predictor. This predictor always guesses that the home team will win, which sets a baseline accuracy of 50% to 60%. The reason for the higher than usual accuracy is due to the idea of home court advantage, which is reflected in "real" games by the home team winning approximately 55% to 62% of the time. In the training data (via the Simple model), this comes out to ~55%.

<img src="eda/graphs/home team win rate.webp">
