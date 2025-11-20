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

## Feature Engineering

I tried a few combinations of features, primarily using the readily-available data. This includes the following stats (at a team level):

-   assists
-   blocks
-   steals
-   fieldGoalsAttempted, fieldGoalsMade, fieldGoalsPercentage
-   threePointersAttempted, threePointersMade, threePointersPercentage
-   freeThrowsAttempted, freeThrowsMade, freeThrowsPercentage
-   reboundsDefensive, reboundsOffensive, reboundsTotal
-   foulsPersonal, turnovers

Using these features didn't result in much, and I think it's due to 2 main reasons. One being that they are all closely tied to eachother (for example, assists are only possible if points were scored). The second issue is that these values are all relative when determining a win/loss. There may be some signal in the raw values (for example, if a team has a high amount of turnovers, they will probably have a harder time winning), but for the most part the values rely on how the other team did in comparison. In theory, a game could be won by scoring a single point, or could still be lost if 100+ points were scored. This means the values have no meaning without context of how the other team did in comparison, which becomes more difficult to include when training on every single game.

This is where I learned about the [Four Factors](https://www.basketball-reference.com/about/factors.html). The Four Factors are measurements determined by analysists that have been shown to have an affect on how well a team performs. The higher the value in each category, (statistically) means there's a higher chance that team will win. The cool thing about them is that they are all relatively independent of each other, but still do a good job of capturing the full picture of the game. They're also represented as a percentage, which fixes the second issue I mentioned earlier, since they can be compared directly across two games with different total scoring.

I chose to use these as my main features, with one "set" of four being for the home team, and the other "set" being for the away team. I chose to use a rolling average of these values for each team over the last 10 games to capture how well the teams have performed recently. This resulted in the following:

<img src="out/pr_curve.png">
<img src="out/roc_curve.png">

Overall, this went well! As seen above, using the Four Factors performed better than the simple model for all models.
(note that for my evaluation, I trained on all regular season games up until this season, and tested on all games during this season)

Some models definitely perform better than others, so I plan on focusing on the better performing models as I incorporate more features. This is also with no hyperparameter tuning, which I plan to do when I fully finish feature engineering.

## Going Forward

For the final report, I have a few ideas I want to try. The main one being I want to implement an [ELO rating](https://en.wikipedia.org/wiki/Elo_rating_system), which is used to "rank" teams by a score. As teams win, their score increases (and vice-versa). ELO does a good job of capturing how "hot" a team is doing as well, as frequent back-to-back wins boosts the score higher. I think using this as a feature will help include some more context to how the teams stack up, further than just how they're doing "on paper".

I also want to look into some player-level features. Currently, everything is done at a team level, which does a good job of capturing how well the team plays together, but it also means it can become inaccurate in the case of roster changes or injuries. Giving more weight on per-player statistics could help account for sudden changes that affect team chemistry. It would be interesting to try to incorporate some kind of information on how many games the team as played with its current roster (or something similar) to capture cases like this.

Lastly, (and somewhat related to above) I want to experiment with the difference between using a team to determine stats (such as ELO) vs finding these stats for each player, then averaging them out for the team. I have a feeling this may capture some things better (such as if a team has a few star players), but may lose information in the form of how well the team works together, so I'm not sure which method would perform better.
