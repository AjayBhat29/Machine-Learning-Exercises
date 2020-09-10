# Restaurant-Review-Sentiment-Analysis
NLP model to perform sentiment analysis based on customer reviews about a restaurant.

# NLP in Python - Step 1
Here the dataset is stored in a .tsv file instead of a .csv file. 

.tsv => tab separated values,  .csv=> comma separated values

This is because the reviews in itself consist of a lot of comma separated words.

While importing the dataset, make sure to specify the tab delimiter

By including the quoting parameter = 3, we are ignoring the double quotes.

# NLP in Python - step 2
This part involves cleaning the texts.

Getting rid of insignificant words like "the" , punctuation marks, etc.

Applying stemming to the imported dataset, this involves removing the 'ing' fomr of words to the verb form(only the root word)
If we do not perform Stemming, then our sparse matrix will be very huge
Matrix containing a lot of zeros is called sparse Matrix. This property is called sparcity. We always should try to reduce sparcity as much as possible.

Converting of all uppercase characters to lowercase characters

# NLP in Python - step 3
This part involves creating the bag of words model

Here we take all unique words from all of our reviews, and create a column(one hot vector) for each word
This process is known as tokenization.

Once the bag of words model is created via tokenization, we apply classification to predict if review is positive(1) or negative(0)

-------------------------------------------------------------------------------------------------------------------------------------
The most common preidction models used for NLP are Naive Bayes and Random Forest Classifiers.

Here we use Naive Bayes model.

Here there is no necessity for applying feature scaling.

For more information on Naive Bayes Algorithm visit the links: 

https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/

https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c

-------------------------------------------------------------------------------------------------------------------------------------

# NLP in Python - Step 4
The last step involves grading of how well our model has wokrd on the validation set.

For this ,we make use of the confusion matrix
Accuracy from confusion matrix is calculates as : (TP + TN)/(TP + TN + FP + FN)

TP = true positive

TN = true negative

FP = false positive

FN = false negative
