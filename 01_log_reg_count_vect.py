import mlflow
from mlflow_utils import get_mlflow_experiment
from mlflow.models.signature import infer_signature

import pandas as pd 
import nltk
from nltk.corpus import stopwords
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from joblib import dump, load
import pickle
import warnings
warnings.filterwarnings("ignore")


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)

def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data

def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data


if __name__ == "__main__":

    # Retreive the mlflow experiment
    experiment = get_mlflow_experiment(experiment_name = "ocp7")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="01_log_reg_count_vect", experiment_id=experiment.experiment_id) as run:

        ## Data reading
        data = pd.read_csv("data_raw/eda-twitter-sentiment-analysis-using-nn.csv", encoding = "ISO-8859-1", engine="python")
        data.columns = ["label", "time", "date", "query", "username", "text"]

        ### Selecting text and label columns
        data=data[['text','label']]

        ### Replacing label '4' with '1'
        data['label'][data['label']==4]=1

        ### Separating positive and negative tweets
        data_pos = data[data['label'] == 1]
        data_neg = data[data['label'] == 0]

        ### Keeping only 25% of the data, to be able to run the program quickly at first
        data_pos = data_pos.iloc[:int(20000)]
        data_neg = data_neg.iloc[:int(20000)]

        ### Merging the two arrays to make an array with 20,000 positive tweets and 20,000 negative tweets
        data = pd.concat([data_pos, data_neg])

        ### Converting uppercase to lowercase
        data['text']=data['text'].str.lower()

        ### Removing English stopwords
        stopwords_list = stopwords.words('english')

        ### Removing stop words
        STOPWORDS = set(stopwords.words('english'))

        data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))

        ### Removing punctuation
        english_punctuations = string.punctuation
        punctuations_list = english_punctuations
        data['text']= data['text'].apply(lambda x: cleaning_punctuations(x))

        ### Removing repeated characters
        data['text'] = data['text'].apply(lambda x: cleaning_repeating_char(x))

        ### Removing emails
        data['text']= data['text'].apply(lambda x: cleaning_email(x))

        ### Removing URLs
        data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))

        ### Removing numbers
        data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))

        ### Tokenization of tweets
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        data['text'] = data['text'].apply(tokenizer.tokenize)

        ### Stemming
        st = nltk.PorterStemmer()
        data['text']= data['text'].apply(lambda x: stemming_on_text(x))

        ### Lemmatization
        lm = nltk.WordNetLemmatizer()
        data['text'] = data['text'].apply(lambda x: lemmatizer_on_text(x))

        ## Bag of Words Approach: Tf-idf

        # Formatting needed for Bow approach
        data['text_string'] = data['text'].apply(lambda x: ' '.join(x))

        # Define X and y
        X = data['text_string']
        y = data['label']

        # Get the signature for mlflow
        signature = infer_signature(X, y)

        # Instanciate count_vectorizer
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)

        # Fit and transform the data
        X = vectorizer.fit_transform(data['text_string'])

        # Separate training and test data (70% - 30%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

        # Make the model with the specified regularization parameter
        log_reg = LogisticRegression(C = 0.0001, random_state=42)
        
        # Log model params to mlflow
        params = log_reg.get_params()
        mlflow.log_params(params)

        # Train on the training data
        log_reg.fit(X_train, y_train)

        # Log the model
        mlflow.sklearn.log_model(log_reg,"LogisticRegression", signature=signature)

        # Make predictions from the test data
        y_pred = log_reg.predict(X_test)

        ### Performance Evaluation

        #### ROC Area Under Curve

        # Calculate and log the ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the ROC curve
        fig_roc = plt.figure()
        roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca(), name='log_reg')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title("ROC curve")
        plt.legend()

        mlflow.log_figure(fig_roc, "roc_curve.png")

        # Set a decision treshold to visualize the confusion matrix
        y_pred = (y_pred > 0.5)

        # Calculate and log Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log the confusion matrix
        fig_cm = plt.figure()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca(), cmap="Greys", colorbar=False)
        plt.title("Confusion Matrix")
        plt.legend()

        mlflow.log_figure(fig_cm, "confusion_matrix.png")

        # Display the rounded result
        print(f"ROC AUC score: {round(roc_auc,2)}")

        # # Save the model
        # dump(log_reg, 'baseline.joblib')

        # Save the vectorizer
        with open('vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        mlflow.log_artifact("vectorizer.pkl")

        # Log a tag to mlflow
        mlflow.set_tags({"model_name": "log_reg_count_vect"})

        # Log a description to mlflow
        mlflow.set_tag("mlflow.note.content", "A binary classifier for text sentiment analysis.")