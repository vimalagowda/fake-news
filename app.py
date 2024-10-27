from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string

app = Flask(__name__)

# Load the model and vectorizer
df_merge = pd.read_csv("D:/Fake.csv/manual_testing.csv")  # Replace "merged_data.csv" with your actual data file
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(df_merge["text"])
y = df_merge["class"]

# Model initialization
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier(random_state=0)
RFC = RandomForestClassifier(random_state=0)

# Fit models
LR.fit(xv_train, y)
DT.fit(xv_train, y)
GBC.fit(xv_train, y)
RFC.fit(xv_train, y)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.get_json()['news']
    news_text = wordopt(news_text)

    new_x_test = vectorization.transform([news_text])
    pred_LR = LR.predict(new_x_test)[0]
    pred_DT = DT.predict(new_x_test)[0]
    pred_GBC = GBC.predict(new_x_test)[0]
    pred_RFC = RFC.predict(new_x_test)[0]

    result = {
        'prediction_LR': int(pred_LR),
        'prediction_DT': int(pred_DT),
        'prediction_GBC':int( pred_GBC),
        'prediction_RFC':int( pred_RFC)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=304)
