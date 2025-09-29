import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the data
data = pd.read_csv('train.csv')  # Change this to your actual file path

# Drop rows with missing 'text' values
data = data.dropna(subset=['text'])

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(), 
    data['label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenization function with reduced max_length
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

# Convert data into a format compatible with the Hugging Face Trainer API
train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize, batched=True)
val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize, batched=True)

# Set the Trainer arguments to save time and resources
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Reduced to 1 epoch
    per_device_train_batch_size=4,  # Smaller batch size
    per_device_eval_batch_size=4,
    warmup_steps=100, 
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_total_limit=1,  # Keep only the last checkpoint
    save_steps=500  # Save after every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model_save_path = "./distilbert_finetuned_fake_news"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")


# model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
# tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
# print('model loaded')