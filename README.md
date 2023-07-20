# Fake Medical News Detection Tool
![Screenshot](https://github.com/changyuhsin1999/Fake_Medical_News_Detection_Tool/blob/main/image/Vitamin-D-in-moderation-protects-against-respiratory-infections-Meta-analysis.jpeg)

As we live in this era with exploding information content, it is always difficult to distinguish whether the information is real or fake. It can be a lot difficult when it comes to medical contents which often include lots of professional terms and acronyms. In this project, I combined two different existing dataset with a mix of real and fake medical journal and news, pass it through TFIDF word vectorizer and feed into trained model for binary classification.

The purpose of this module is to let people copy and paste the medical news that they found and do a quick fact check to see whether it is reliable to trust the sources.

## Data
1. [Kaggle - COVID Real News](https://www.kaggle.com/datasets/arashnic/covid19-fake-news?select=NewsRealCOVID-19_7.csv) for real news
2. [A comprehensive data repository for fake health news - dataset/content/HealthRelease](https://github.com/EnyanDai/FakeHealth/tree/master/dataset/content/HealthRelease) for fake news

I combined both datasets and shuffle them, get rid of null values and unneccesary columns to get raw_df.csv in the data folder

## Visual Interface
![Screenshot](https://github.com/changyuhsin1999/Fake_Medical_News_Detection_Tool/blob/main/image/Screen%20Shot%202023-07-20%20at%202.25.39%20PM.png)
