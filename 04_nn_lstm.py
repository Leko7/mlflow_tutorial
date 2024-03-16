import mlflow
from mlflow_utils import get_mlflow_experiment
from mlflow.models.signature import infer_signature

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Removal of unnecessary imports and warnings configuration
import warnings
warnings.filterwarnings("ignore")


def cleaning_stopwords(text):
    """
    Removes stopwords from the text.
    """
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def cleaning_punctuations(text):
    """
    Removes punctuation from the text.
    """
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    """
    Replaces repeating characters in the text.
    """
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_email(data):
    """
    Removes email addresses from the text.
    """
    return re.sub('@[^\s]+', ' ', data)

def cleaning_URLs(data):
    """
    Removes URLs from the text.
    """
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

def cleaning_numbers(data):
    """
    Removes numbers from the text.
    """
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data):
    """
    Applies stemming to the text.
    """
    text = [st.stem(word) for word in data]
    return data

def lemmatizer_on_text(data):
    """
    Applies lemmatization to the text.
    """
    text = [lm.lemmatize(word) for word in data]
    return data

def get_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig("confusion_matrix.png")  # Save the figure
    plt.close()

if __name__ == "__main__":

    # Retreive the mlflow experiment
    experiment = get_mlflow_experiment(experiment_name = "ocp7")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="04_nn_lstm", experiment_id=experiment.experiment_id) as run:

        # Data reading
        data = pd.read_csv("data_raw/eda-twitter-sentiment-analysis-using-nn.csv", encoding = "ISO-8859-1", engine="python")
        data.columns = ["label", "time", "date", "query", "username", "text"]

        # Selecting text and label columns
        data=data[['text','label']]

        # Replacing label '4' with '1'
        data['label'][data['label']==4]=1

        # Separating positive and negative tweets
        data_pos = data[data['label'] == 1]
        data_neg = data[data['label'] == 0]

        # Keeping only 25% of the data for quick run
        data_pos = data_pos.iloc[:int(20000)]
        data_neg = data_neg.iloc[:int(20000)]

        # # Keeping only 25% of the data for quick run
        # data_pos = data_pos.iloc[:int(10)]
        # data_neg = data_neg.iloc[:int(10)]

        # Merging positive and negative tweets
        data = pd.concat([data_pos, data_neg])

        # Converting uppercase to lowercase
        data['text']=data['text'].str.lower()

        # Removing English stopwords
        STOPWORDS = set(stopwords.words('english'))

        # Removing stop words
        data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))

        # Removing punctuation
        punctuations_list = string.punctuation
        data['text']= data['text'].apply(lambda x: cleaning_punctuations(x))

        # Removing repeated characters
        data['text'] = data['text'].apply(lambda x: cleaning_repeating_char(x))

        # Removing emails
        data['text']= data['text'].apply(lambda x: cleaning_email(x))

        # Removing URLs
        data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))

        # Removing numbers
        data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))

        # Tokenization of tweets
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        data['text'] = data['text'].apply(tokenizer.tokenize)

        # Stemming
        st = nltk.PorterStemmer()
        data['text']= data['text'].apply(lambda x: stemming_on_text(x))

        # Lemmatization
        lm = nltk.WordNetLemmatizer()
        data['text'] = data['text'].apply(lambda x: lemmatizer_on_text(x))

        # Sequential encoding of tokens
        max_len = 500
        tok = Tokenizer(num_words=2000)
        tok.fit_on_texts(data.text)
        sequences = tok.texts_to_sequences(data.text)
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

        # Splitting data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, data.label, test_size=0.3, random_state=2)

        # Model compilation
        def tensorflow_based_baseline_model():
            inputs = Input(name='inputs', shape=[max_len])
            layer = Embedding(2000, 50, input_length=max_len)(inputs)
            layer = LSTM(64)(layer)
            layer = Dense(256, name='FC1')(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.5)(layer)
            layer = Dense(1, name='out_layer')(layer)
            layer = Activation('sigmoid')(layer)
            model = Model(inputs=inputs, outputs=layer)
            return model

        # Model instantiation and configuration
        #auc_roc = AUC(curve='ROC')
        model = tensorflow_based_baseline_model()
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['Accuracy'])

        # Model training
        history = model.fit(X_train, Y_train, batch_size=80, epochs=6, validation_split=0.1)
        print('Training finished !!')

        # Model testing
        accr1 = model.evaluate(X_test, Y_test)
        print('Test set\n  accuracy: {:0.2f}'.format(accr1[1]))

        # Predict probabilities for ROC Curve and ROC AUC metric
        pred_prob = model.predict(X_test)

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(Y_test, pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()


        class_names = ['negative', 'positive']
        y_pred = (pred_prob > 0.5)
        # Get the confusion matrix
        cm = confusion_matrix(Y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        get_confusion_matrix(df_cm)

        # # Saves the architecture, weights, and training config of the model
        # model.save('model_full.h5')  
        # print("Model saved to 'model_full.h5'")

        # # Save tokenizer
        # with open('tokenizer.pickle', 'wb') as handle:
        #     pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print("Tokenizer saved to 'tokenizer.pickle'")

        # Mlflow logging
        params = {
            "batch_size": 80,
            "epochs": 6
        }
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accr1[1])
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_artifact("roc_curve.png")
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact('tokenizer.pickle')
        mlflow.tensorflow.log_model(model, "model")
        

        mlflow.set_tags({"model_name": "nn_lstm"})

        # Log a description to mlflow
        mlflow.set_tag("mlflow.note.content", "A binary classifier for text sentiment analysis.")