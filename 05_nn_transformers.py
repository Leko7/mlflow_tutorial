import mlflow
from mlflow_utils import get_mlflow_experiment
from mlflow.models.signature import infer_signature

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from itertools import cycle
from collections import defaultdict
from textwrap import wrap

# Torch ML libraries
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Misc.
import warnings
warnings.filterwarnings('ignore')

# Set intial variables and constants
# %config InlineBackend.figure_format='retina'

# Function to convert score to sentiment
def to_sentiment(rating):

    rating = int(rating)

    # Convert to class
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get outouts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values

# Function for a single training iteration
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # Backward prop
        loss.backward()

        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get model ouptuts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig("confusion_matrix.png")  # Save the figure
    plt.close()

def plot_roc_curve(y_true, y_score):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_auc_curve.png")  # Save the figure
    plt.close()

class GPReviewDataset(Dataset):
    # Constructor Function
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Length magic method
    def __len__(self):
        return len(self.reviews)

    # get item magic method
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        # Encoded format to be returned
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True, # Trying this
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
    
# Build the Sentiment Classifier class
class SentimentClassifier(nn.Module):

    # Constructor class
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # # Forward propagaion class
    # def forward(self, input_ids, attention_mask):
    #     _, pooled_output = self.bert(
    #       input_ids=input_ids,
    #       attention_mask=attention_mask
    #     )
    #     #  Add a dropout layer
    #     output = self.drop(pooled_output)
    #     return self.out(output)

    # Debug suggestion by a kaggle user in the comments
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = bert_output.pooler_output # Accessing the pooled output
        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)
    
if __name__ == "__main__":

    # Retreive the mlflow experiment
    experiment = get_mlflow_experiment(experiment_name = "ocp7")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="05_nn_transformers", experiment_id=experiment.experiment_id) as run:

        # Graph Designs
        sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
        HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
        sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
        rcParams['figure.figsize'] = 12, 8

        # Random seed for reproducibilty
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        # Set GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
            # Set PyTorch to use the GPU
            torch.cuda.set_device(0)
        else:
            print("CUDA is not available. Check your installation.")
        

        df = pd.read_csv("data_raw/eda-twitter-sentiment-analysis-using-nn.csv", encoding = "ISO-8859-1", engine="python")
        df.columns = ["label", "time", "date", "query", "username", "text"]
        df=df[['text','label']]

        df_pos = df[df['label'] == 4]
        df_neg = df[df['label'] == 0]

        print("number of positives", df_pos.shape[0])
        print("number of negatives", df_neg.shape[0])

        # df_pos = df_pos.iloc[:int(20000)]
        # df_neg = df_neg.iloc[:int(20000)]

        # df_pos = df_pos.iloc[:int(8000)]
        # df_neg = df_neg.iloc[:int(8000)]

        df_pos = df_pos.iloc[:int(500)]
        df_neg = df_neg.iloc[:int(500)]


        df = pd.concat([df_pos, df_neg])


        df['sentiment'] = df['label'].replace(4, 1)

        class_names = ['negative', 'positive']

        # Set the model name
        MODEL_NAME = 'bert-base-cased'

        # Build a BERT based tokenizer
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        MAX_LEN = 150
            
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


        # Create train, test and val data loaders
        BATCH_SIZE = 16
        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

        # Load the basic BERT model
        bert_model = BertModel.from_pretrained(MODEL_NAME)
            
        # Instantiate the model and move to classifier
        model = SentimentClassifier(len(class_names))
        model = model.to(device)

        # Number of iterations
        EPOCHS = 6

        # Optimizer Adam
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Set the loss function
        loss_fn = nn.CrossEntropyLoss().to(device)

        # %%time

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(EPOCHS):

            # Show details
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print("-" * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)
            )

            print(f"Train loss {train_loss} accuracy {train_acc}")

            # Get model performance (accuracy and loss)
            val_acc, val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val)
            )

            print(f"Val   loss {val_loss} accuracy {val_acc}")
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            # If we beat prev performance
            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc

        test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
        )

        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model,
            test_data_loader
        )

        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        get_confusion_matrix(df_cm)

        # Additional code for roc_auc metric and roc curve
        positive_class_probs = y_pred_probs[:, 1]  # Adjust index if necessary based on your model's output
        roc_auc = roc_auc_score(y_test, positive_class_probs)
        print(f"ROC AUC Score: {roc_auc}")
        plot_roc_curve(y_test, positive_class_probs)

        # Mlflow logging

        params = {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": 2e-5,
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN
        }
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_metric("roc_auc", roc_auc)

        signature = infer_signature(df['text'], df['sentiment'])
        mlflow.pytorch.log_model(model, "model", signature=signature)

        # Log a figure to mlflow
        #mlflow.log_figure(cm, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("roc_auc_curve.png")

        # Log a tag to mlflow
        mlflow.set_tags({"model_name": "nn_transformers"})

        # Log a description to mlflow
        mlflow.set_tag("mlflow.note.content", "A binary classifier for text sentiment analysis.")
