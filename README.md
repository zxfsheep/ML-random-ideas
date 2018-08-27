## Machine learning random ideas

Place to drop new ideas that I came up with while practicing data science and machine learning. Some of them are linked to Jupyter notebooks with more details and implementation.

[My Index page for all repositories.](https://github.com/zxfsheep/Index/blob/master/README.md)

---
#### 1. [A simple way to find a good cross validation split](https://github.com/zxfsheep/ML-random-ideas/blob/master/Find_best_split.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;One seemingly minute issue during cross validation is how to choose the random split. One can choose the lucky number as random seed, or a bunch of numbers. I found a simple trick to quantitatively choose a decent split.
   
#### 2. [Dealing with fragmented and nonuniform time series data](https://github.com/zxfsheep/ML-random-ideas/blob/master/Dealing_with_messy_time_series_data.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are classical effective time series models that deal with both seasonal and nonseasonal data, such as **ARMA**(autoregressive–moving-average) and **ARIMA**(autoregressive integrated moving average). In practice, it might be hard to apply these standard models, due to different objectives and more seriously, very fragmented and nonuniform times series data. I came up with a very simple way to extract features out of such messy situations.
   
#### 3. Feature elimination
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Very often after initial feature engineering, for example using [the trick above](https://github.com/zxfsheep/ML-random-ideas/blob/master/README.md#2-dealing-with-fragmented-and-nonuniform-time-series-data), I might get too many features, more than desired. There are at least two types of unwanted features:
  * features that never seem useful;
  * features that seems useful sometimes, but it might be overfitting to the training data.
  
Since most machine learning models can also output the feature importances in various ways, we can utilize those information to eliminate some features, which both speeds up future trainings and also avoids overfitting. 

Initially I just remove features with lowest average importance, which correspond to the first type. It turns out this does not help that much except speeding up training, since models such as gradient boosting trees will automatically focus on useful features. The second type of unwanted features have bigger impact on prediction quality. To reduce overfitting, one idea that I learned from Kaggle is to **randomly permute the labels** and look at how the feature importances change. Ones that do not change much are probably noises that cause overfitting. However it is difficult to use in practice, as a lot of permutations are needed to make meaningful conclusions. When each training takes a long time, this is very wasteful. These extra trainings are otherwise completely useless since the labels are incorrect.

In practice, I came up with the following selection scheme that worked well: examine the feature importances across the folds in cross validation. Instead of mean importance, look at minimum (and possibly variance) of importances. In addition to features with low mean importance, I also remove features that have low (say 0) **min importance**, even if they have higher mean importance, because this indicates that they might be overfitting to some folds. This scheme works better when the training dataset is large, and we can take a larger number of folds to eliminate more overfitting features.

#### 4. [Ranking encoding](https://github.com/zxfsheep/ML-random-ideas/blob/master/Ranking_encoding.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A common technique to deal with categorical features for non-deep machine learning models is one-hot encoding(For deep learning, they can be embedded in a vector space as in **Factorization Machines**). However, this has some drawbacks in practice. For example, this creates a lot of sparse features if the number of categories is large, which can confuse the machine learning models. Another popular encoding is **mean encoding**, i.e. encode a category by its target mean. This however can still cause potential trouble, as categories with the same or very close target mean can be mixed together, but we still want to differentiate them as they might have very different interactions with other features. I decided to encode all categorical features by their **ranking** of target mean, which is also a very nice application of pandas.
