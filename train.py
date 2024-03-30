import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset


class CustomRegressionModel(torch.nn.Module):
    """
    A custom PyTorch regression model that wraps a pretrained transformer model
    with an added linear regression head and an activation function.

    Parameters:
    - base_model (transformers.PreTrainedModel): The pretrained transformer model.

    Attributes:
    - base_model: The underlying pretrained model.
    - regression_head: A linear layer for regression.
    - activation: An activation function to ensure output is between -1 and 1.
    """

    def __init__(self, base_model):
        super(CustomRegressionModel, self).__init__()
        self.base_model = base_model
        self.regression_head = torch.nn.Linear(768, 1)
        self.activation = torch.nn.Tanh()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the model.

        Parameters:
        - input_ids (torch.Tensor): Tokens' input IDs.
        - attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
        - labels (Optional[torch.Tensor]): Labels for the input data; not used in forward.

        Returns:
        - torch.Tensor: The model's predictions.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[1]
        logits = self.regression_head(sequence_output)
        logits = self.activation(logits)
        return logits.view(-1, 1)

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
        - torch.device: The device on which the model is.
        """
        return next(self.parameters()).device


class RegressionDataset(Dataset):
    """
    A PyTorch Dataset for regression tasks.

    Parameters:
    - encodings (dict): Encoded text data for the inputs.
    - labels (list): List of labels for the input data.

    Attributes:
    - encodings: The encoded text data.
    - labels: The labels.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves an item by index.

        Parameters:
        - idx (int): The index of the item.

        Returns:
        - dict: The encoded input and its label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]], dtype=torch.float)
        return item

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: The number of items in the dataset.
        """
        return len(self.labels)


class CustomTrainer(Trainer):
    """
    Custom trainer class for regression tasks, extending Hugging Face's Trainer.

    This class overrides the compute_loss method to handle regression-specific loss computation.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss using Mean Squared Error (MSE) for regression.

        Parameters:
        - model (torch.nn.Module): The model to compute loss for.
        - inputs (dict): Inputs to the model.
        - return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
        - torch.Tensor: The computed loss.
        - torch.Tensor (optional): The model outputs if return_outputs is True.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs
        labels = labels.float().view(-1, 1)
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    # Load a pretrained model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Initialize the custom model
    model = CustomRegressionModel(base_model)
    # Load the dataset
    data_path = 'data/Tweets.csv'
    df = pd.read_csv(data_path).iloc[:]


    X = df['text']
    y = df['airline_sentiment']

    # Convert sentiments to numeric and tokenize texts
    sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['airline_sentiment'] = df['airline_sentiment'].map(sentiment_mapping).astype(float)
    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['airline_sentiment'], test_size=0.2)

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=128)

    train_dataset = RegressionDataset(train_encodings, y_train.tolist())
    val_dataset = RegressionDataset(val_encodings, y_val.tolist())

    training_args = TrainingArguments(
        output_dir='./results',  # output directory for model and logs
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",  # perform evaluation each epoch
        save_strategy="epoch",  # save the model every epoch
        load_best_model_at_end=True,  # load the best model when finished training
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None  # Define a function for compute_metrics if you need to evaluate specific metrics
    )

    trainer.train()
    model_path = './results/trained_roberta_regression'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
