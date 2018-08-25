## Machine learning random ideas

Place to drop new ideas that I came up with while practicing data science and machine learning. 

[My Index page for all repositories.](https://github.com/zxfsheep/Index/blob/master/README.md)

---
### 1. Simple way to find a good cross validation split
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;One seemingly minute issue during cross validation is how to choose the random split. One can choose the lucky number as random seed, or a bunch of numbers. I found a simple trick to quantitatively choose a decent split.
   
### 2. Dealing with fragmented and nonuniform time series data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are classical effective time series models that deal with both seasonal and nonseasonal data, such as ARMA(autoregressive–moving-average) and ARIMA(autoregressive integrated moving average). In practice, it might be hard to apply these standard models, due to different objectives and more seriously, very fragmented and nonuniform times series data. I came up with a very simple way to extract features out of such messy situations.
   
### 3. Ranking encoding
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A common technique to deal with categorical feature is one-hot encoding. However, this has some drawbacks in practice. For example, this creates a lot of sparse features if the number of categories is large, which can confuse the machine learning models. Another popular encoding is mean encoding, i.e. encode a category by its target mean. This however can still cause potential trouble, as categories with the same or very close target mean can be mixed together, but we still want to differentiate them as they might have very different interactions with other features. I decided to encode all categorical features by their ranking of target mean, which is also a very nice application of pandas.
