import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import joblib


stop_words = set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


dataset = pd.read_csv(r"C:\Users\dipak\OneDrive\Desktop\Full stack data science\Projects\Mini project\amazon_review_full_csv\train.csv")

dataset.drop(dataset.columns[[1]], axis=1, inplace=True)
dataset.rename(columns={'All': 'Review', '2': 'Rating'}, inplace=True)

dataset.head()
dataset.Rating.value_counts(normalize = True)



#**********PREPROCESING THE DATA**************
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words).strip()


dataset['Review'] = dataset['Review'].apply(preprocess_text)


#**********STEMING************

def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


dataset['Review'] = dataset['Review'].apply(stem_text)



count = Counter(' '.join(dataset['Review']).split())

words = pd.DataFrame(count.items(), columns=['Words', 'Frequency'])

words = words.sort_values('Frequency', ascending=False).reset_index(drop=True)

words['Rank'] = words.index + 1
words = words[['Rank', 'Words', 'Frequency']]

#&***********WordCloud***********

def generate_wordcloud(input):
    cloud = WordCloud(width=1500, height=800, max_words=500, background_color='black', colormap='coolwarm')
    wordcloud = cloud.generate(input)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

positive_words = " ".join(dataset[dataset['Rating'] == 2]['Review'])#SHOWS THE POSITIVE WORDS
generate_wordcloud(positive_words)

negative_words = " ".join(dataset[dataset['Rating'] == 1]['Review'])#SHOWAW NEGATICVE WORDS
generate_wordcloud(negative_words)


#********Trainin the model ************

vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1,2))

X_train = vectorizer.fit_transform(dataset['Review'])
y_train = dataset['Rating']

from joblib import dump

joblib.dump(vectorizer, open(r"C:\Users\dipak\OneDrive\Desktop\Full stack data science\Projects\Mini project\amazon_review_full_csv\vectorizer.joblib",'wb'))


'''
#Support Vrctot Machine

clf = LinearSVC()


LinearSVC(X_train,y_train)

clf.fit(X_train, y_train)




#naive

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#logistic regression


#Random Forest
from sklearn.ensemble import RandomForestRegressor

Rfr = RandomForestRegressor()
Rfr.fit(X_train, y_train)

#DecisionTree Regressor
from sklearn.tree import DecisionTreeRegressor

decision  = DecisionTreeRegressor()
decision.fit(X_train,y_train)


from sklearn.linear_model import LogisticRegression

model  = LogisticRegression()
model.fit(X_train,y_train)



pickle.dump(logistic, open(r'C:\Users\dipak\OneDrive\Desktop\Full stack data science\Projects\Mini project\amazon_review_full_csv\model.pkl', 'wb'))

'''

from sklearn.linear_model import LogisticRegression

logistic  = LogisticRegression()
logistic.fit(X_train,y_train)

from joblib import dump

joblib.dump(logistic, open(r'C:\Users\dipak\OneDrive\Desktop\Full stack data science\Projects\Mini project\amazon_review_full_csv\model.joblib', 'wb'))

#Tesing phase :::::::::::
    
test =pd.read_csv(r'C:\Users\dipak\OneDrive\Desktop\Full stack data science\Projects\Mini project\amazon_review_full_csv\test.csv',nrows = 40000,header=None)
    
test.columns=['Rating','Title','Review']
test = test[['Rating', 'Review']].reset_index(drop=True)
                    
test['Review'] = test['Review'].apply(preprocess_text)
test['Review'] = test['Review'].apply(stem_text)

X_test = vectorizer.transform(test['Review'])
y_test = test['Rating']



pickled_model = pickle.load(open('model.pkl', 'rb'))

y_pred = pickled_model.predict(X_test)





#Testing the model
score = accuracy_score(y_test,y_pred)
print(score)



# last step
def predict_sentiment(text):
    preprocessed_text = stem_text(preprocess_text(text))
    
    features = vectorizer.transform([preprocessed_text])
    
    prediction = pickled_model.predict(features)[0]
    
    if prediction == 1:
        return "Negative"
    else:
        return "Positive"
    
    
    
#Trial
print(predict_sentiment("this is not good roduct"))
print(predict_sentiment("this is bad roduct"))
print(predict_sentiment("this is fuching roduct"))

print(predict_sentiment("it is a bad product and a percect for college students"))
