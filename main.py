import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("C:\\Users\\Akash Chintaluri\\PycharmProjects\\Language Detector\\data\\LanguageDetection.csv")

x = np.array(data["Text"])
y = np.array(data["Language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
model_score = model.score(X_test, y_test)
print("Welcome to Language Detector")

while True:
    print("----------------------------------------------")
    user_input = input("Enter a Text: ")
    input_transformed = cv.transform([user_input]).toarray()

    if not input_transformed.any():
        print("Sorry, the text entered cannot be recognized.")
    else:
        output = model.predict(input_transformed)
        print(output)

    n = input("Enter 'q' to quit, or any other key to continue detecting: ")
    if n == 'q':
        break
