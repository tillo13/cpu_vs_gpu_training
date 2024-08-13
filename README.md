# Yelp Polarity Dataset Sentiment Analysis  
  
This repository contains code for training and evaluating a sentiment analysis model using the Yelp Polarity dataset. The code leverages the `transformers` library from Hugging Face and evaluates the model using various metrics.  
  
## Setup  
  
### Requirements  
  
- Python 3.7+  
- `torch`  
- `transformers`  
- `datasets`  
- `numpy`  
- `matplotlib`  
- `psutil`  
- `gputil`  
- `scikit-learn`  
  
### Install Dependencies  
  
You can install the required Python packages using pip:  
  
```bash  
pip install torch transformers datasets numpy matplotlib psutil gputil scikit-learn  
 

Usage
 

System Information:

The script gathers system information including CPU and GPU details.
Data Loading:

The script loads the Yelp Polarity dataset and creates a smaller subset for faster training and testing.
Model and Tokenizer:

The script loads a pre-trained DistilBERT model and its tokenizer from Hugging Face.
Data Preprocessing:

The text data is tokenized using the loaded tokenizer.
Training:

The script trains the model on the tokenized dataset and measures the training time.
Evaluation:

The script evaluates both the pre-trained and fine-tuned models on a test subset.
Metrics and Plots:

The script calculates various metrics (accuracy, precision, recall, F1-score) and generates plots for training/evaluation loss and the confusion matrix.
Results Storage:

The script stores system information, training time, and evaluation metrics in a CSV file.
Run the Script
 
To run the script, simply execute:

python your_script_name.py  
 

Output
 
The script produces the following output files:
training_evaluation_loss.png: A plot of the training and evaluation loss.
confusion_matrix.png: A plot of the confusion matrix.
training_comparison.csv: A CSV file containing system information, training time, and evaluation metrics.
Example
 
Here's an example of the output you can expect:
CSV Output:


computer,system,release,version,machine,processor,cpu_count,cpu_freq_max,ram_total,gpu_name,gpu_load,gpu_memory_total,gpu_memory_used,gpu_temperature,device,total_training_time_seconds,epoch_times_seconds,pretrained_accuracy,fine_tuned_accuracy,pretrained_f1,fine_tuned_f1,training_loss,evaluation_loss  
my-computer,Windows,10,10.0.19041,AMD64,Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz,8,1992.0,15.9,NVIDIA GeForce GTX 1050,30.0,4096,2048,70.0,GPU,360.0,[120.0, 120.0, 120.0],0.85,0.95,0.80,0.90,0.2,0.15  
 

License
 
This project is licensed under the MIT License.
