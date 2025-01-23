from transformers import Trainer
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()


class StarModel(Trainer):
    def __init__(
        self,
        model_name="bert-base-uncased",
        output_dir="./starclass_bert",
        num_train_epochs=5,
        learning_rate=5e-5,
        batch_size=16,
        max_length=128,
        hf_token=os.getenv("HF_TOKEN"),
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        if hf_token:
            login(hf_token)
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, csv_file, text_column="sentence", label_column="label"):
        df = pd.read_csv(csv_file)
        df[label_column] = self.label_encoder.fit_transform(df[label_column])
        # Split Data
        train_text, val_text, train_labels, val_labels = train_test_split(
            df[text_column].tolist(),
            df[label_column].tolist(),
            test_size=0.2,
            random_state=42,
        )
        # Create Datasets
        self.train_data = Dataset.from_dict(
            {"sentence": train_text, "label": train_labels}
        )
        self.val_data = Dataset.from_dict({"sentence": val_text, "label": val_labels})

    def initialize_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.tokenize_data()
        num_labels = len(self.label_encoder.classes_)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def tokenize_data(self):
        def preprocess(examples):
            return self.tokenizer(
                examples["sentence"],
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

        self.train_data = self.train_data.map(preprocess, batched=True)
        self.val_data = self.val_data.map(preprocess, batched=True)

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def setup_trainer(self):
        # push_to_hub=True,
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            optim="adamw_torch_fused",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()

    def save_model(self, model_path="./fine_tuned_model"):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def push_to_hub(self):
        self.trainer.create_model_card()
        self.trainer.push_to_hub()

    def show_metrics(self):
        predictions = self.trainer.predict(self.val_data)
        logits = predictions.predictions
        predicted_labels = np.argmax(logits, axis=1)
        true_labels = np.array(self.val_data["label"])

        cm = confusion_matrix(true_labels, predicted_labels)
        label_names = list(self.label_encoder.classes_)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

        # Print classification report
        print("Classification Report:")
        print(
            classification_report(
                true_labels, predicted_labels, target_names=label_names
            )
        )

    # Cross entropy loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    # Initalize Trainer
    starModel = StarModel()
    starModel.preprocess_data("data/exported_df_no_outliers.csv")
    starModel.initialize_model()
    starModel.setup_trainer()
    starModel.train()
    starModel.save_model()
    starModel.push_to_hub()
    starModel.show_metrics()
