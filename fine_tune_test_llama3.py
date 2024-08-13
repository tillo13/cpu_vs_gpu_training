import os  
import time  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay  
from datasets import load_dataset  
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction  
  
# Load the full dataset  
print("Loading dataset...")  
dataset = load_dataset('yelp_polarity', split='train')  
  
# Create a smaller subset (e.g., 2,000 samples) for faster training  
subset_size = 2000  
print(f"Selecting the first {subset_size} samples from the dataset...")  
small_dataset = dataset.select(range(subset_size))  
  
# Load the Llama 3 model and tokenizer  
model_name = "meta-llama3-8b"  # Adjust the model name as necessary  
print(f"Loading model and tokenizer for {model_name}...")  
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
  
# Preprocess the data  
def preprocess_function(examples):  
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)  
  
print("Tokenizing the dataset...")  
tokenized_datasets = small_dataset.map(preprocess_function, batched=True)  
  
# Define a compute_metrics function  
def compute_metrics(p: EvalPrediction):  
    preds = np.argmax(p.predictions, axis=1)  
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')  
    acc = accuracy_score(p.label_ids, preds)  
    return {  
        'accuracy': acc,  
        'precision': precision,  
        'recall': recall,  
        'f1': f1,  
    }  
  
# Define training arguments  
print("Setting up training arguments...")  
training_args = TrainingArguments(  
    output_dir="./results",  
    eval_strategy="steps",  
    eval_steps=50,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    num_train_epochs=1,  # Reduced epochs for faster training  
    logging_dir='./logs',  
    logging_steps=50,  
    save_total_limit=2,  
    save_steps=500,  
    use_cpu=True  
)  
  
# Initialize the Trainer  
print("Initializing the Trainer...")  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=tokenized_datasets,  
    eval_dataset=tokenized_datasets,  
    compute_metrics=compute_metrics,  
)  
  
# Measure training time and train the model  
print("Starting training...")  
start_time = time.time()  
  
# Training with progress and estimated time of completion  
step_times = []  
for step in range(training_args.num_train_epochs):  
    epoch_start = time.time()  
    train_result = trainer.train()  
    epoch_end = time.time()  
  
    step_time = epoch_end - epoch_start  
    step_times.append(step_time)  
  
    avg_step_time = np.mean(step_times)  
    remaining_steps = training_args.num_train_epochs - (step + 1)  
    estimated_time_remaining = avg_step_time * remaining_steps  
  
    print(f"Epoch {step + 1}/{training_args.num_train_epochs} completed in {step_time:.2f} seconds.")  
    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds.")  
  
end_time = time.time()  
training_time = end_time - start_time  
  
print(f"Total training time: {training_time:.2f} seconds")  
  
# Load the test dataset  
print("Loading test dataset...")  
test_dataset = load_dataset('yelp_polarity', split='test')  
  
# Create a smaller subset for quick testing  
test_subset_size = 2000  
print(f"Selecting the first {test_subset_size} samples from the test dataset...")  
small_test_dataset = test_dataset.select(range(test_subset_size))  
  
# Tokenize the test dataset  
print("Tokenizing the test dataset...")  
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)  
  
# Evaluate the fine-tuned model  
print("Evaluating the fine-tuned model...")  
fine_tuned_eval_result = trainer.evaluate(tokenized_test_dataset)  
print(f"Fine-tuned Model Evaluation Metrics: {fine_tuned_eval_result}")  
  
# Print evaluation metrics  
print("Fine-tuned Model Evaluation Metrics:")  
for key, value in fine_tuned_eval_result.items():  
    print(f"{key}: {value:.4f}")  
  
# Plot training and evaluation loss and save as PNG  
print("Plotting training and evaluation loss...")  
  
# Collect training and evaluation loss  
train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]  
eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]  
  
plt.figure(figsize=(10, 5))  
plt.plot(range(len(train_loss)), train_loss, label='Training Loss')  
plt.plot(range(len(eval_loss)), eval_loss, label='Evaluation Loss')  
plt.xlabel('Steps')  
plt.ylabel('Loss')  
plt.legend()  
plt.title('Training and Evaluation Loss')  
plt.savefig('training_evaluation_loss.png')  
plt.close()  
  
# Generate confusion matrix and save as PNG  
print("Generating confusion matrix...")  
  
# Make predictions on the test dataset  
predictions = trainer.predict(tokenized_test_dataset).predictions  
pred_labels = np.argmax(predictions, axis=1)  
true_labels = tokenized_test_dataset['label']  
  
# Compute the confusion matrix  
cm = confusion_matrix(true_labels, pred_labels)  
  
# Plot the confusion matrix  
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  
disp.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix')  
plt.savefig('confusion_matrix.png')  
plt.close()  
  
print("All done!")  
