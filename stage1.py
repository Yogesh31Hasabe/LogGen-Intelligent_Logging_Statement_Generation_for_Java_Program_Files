import os
import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
from scipy.spatial.distance import cdist
import numpy as np

# Define parameters
model_name = 'meta-llama/Llama-3.2-1B'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '../dataset/train'
output_dir = './output/predictions'
model_dir = './models/stage1/lama_finetuned_stage1_model'
tokenizer_dir = './models/stage1/lama_finetuned_stage1_tokenizer'
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
def setup():
    model = ""
    tokenizer = ""
    if os.path.exists(model_dir) and os.path.exists(tokenizer_dir):
        print(f"Loading pre-trained model and tokenizer from {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        print("Model or tokenizer not found. Fine-tuning the model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        def load_dataset_from_csvs(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            datasets = []
            for file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(data_dir, file))
                    dataset = Dataset.from_pandas(df)
                    datasets.append(dataset)
                except pd.errors.ParserError as e:
                    print(f"Error processing file {file}: {e}")
                    continue
            return concatenate_datasets(datasets)

        dataset = load_dataset_from_csvs(data_dir)

        clean_pattern = r"(//.*?$)|(/\*.*?\*/)|(/\*\*.*?\*/)|(^\s*import\s.*?;$)"
        max_length = 1024
        
        def preprocess_function(examples):
            global max_length
        
            content = examples["Input"]
            cleaned_text = re.sub(clean_pattern, '', content, flags=re.DOTALL | re.MULTILINE)
            split_pattern = r'(?<=[;{})])\s*(?=\n|$)'
            
            sentences = [sentence.strip() for sentence in re.split(split_pattern, cleaned_text) if sentence.strip()]

            
            tokenized_sentences = [tokenizer(sentence, add_special_tokens=False) for sentence in sentences]

            
            input_ids = []
            attention_mask = []
            labels = []
            
            labels_list = [int(label) for label in examples["Label"].split()]
            modified_labels_list = []

            for i,tokenized_sentence in enumerate(tokenized_sentences):
                input_ids.extend(tokenized_sentence["input_ids"])
                attention_mask.extend(tokenized_sentence["attention_mask"])
                labels.extend([labels_list[i]] * len(tokenized_sentence["input_ids"]))
            
            
            input_ids = input_ids
            attention_mask = attention_mask
            input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(input_ids))
            attention_mask = attention_mask[:max_length] + [0] * max(0, max_length - len(attention_mask))
            labels = labels[:max_length] + [0] * max(0, max_length - len(labels))
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels  
            }

        
        processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

        train_test_split = processed_dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        
        print("Processed Train dataset columns:", train_dataset.column_names)
        print("Processed Eval dataset columns:", eval_dataset.column_names)

        
        training_args = TrainingArguments(
            output_dir=model_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            save_steps=10_000,
            save_total_limit=2,
            remove_unused_columns=True,
        )

        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        torch.cuda.empty_cache()
        trainer.train()

        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Fine-tuned model and tokenizer saved to {model_dir} and {tokenizer_dir}")   
    return model,tokenizer


def predict_log_positions(input_text, model, tokenizer, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    
    model.to(device)

    
    with torch.no_grad():
        
        outputs = model(**inputs)
        logits = outputs.logits

        probabilities = torch.sigmoid(logits).squeeze().cpu().tolist()
        flat_probabilities = [prob for sublist in probabilities for prob in sublist]
    
    high_threshold = 0.9  
    mid_threshold = 0.7   

    predictions = []
    for i, prob in enumerate(flat_probabilities):
        if prob > high_threshold:
            predictions.append(1)
        elif prob > mid_threshold:
            neighboring_probs = flat_probabilities[max(0, i - 1):min(len(flat_probabilities), i + 2)]
            if sum(1 for p in neighboring_probs if p > high_threshold) > 1:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(0)

    
    log_positions = []
    added_lines = set()
    buffer_distance = 2  

    for i, label in enumerate(predictions):
        if label == 1:
            line_number = input_text.count('\n', 0, i) + 1
            
            if all(abs(line_number - prev_line) > buffer_distance for prev_line in added_lines):
                log_positions.append(line_number)
                added_lines.add(line_number)

    return log_positions    


def evaluate_model_accuracy_with_distance(csv_file_path, model, tokenizer, device):
    df = pd.read_csv(csv_file_path)

    total_log_lines = 0
    correct_predictions = 0
    predicted_positions_all = []
    true_log_positions_all = []

    clean_pattern = r"(//.*?$)|(/\*.*?\*/)|(/\*\*.*?\*/)|(^\s*import\s.*?;$)"
    split_pattern = r'(?<=[;})])\s*(?=\n|$)'

    for _, row in df.iterrows():
        
        content = row["Input"]
        labels = [int(label) for label in row["Label"].split()]

        if 1 not in labels: 
            continue

        
        cleaned_content = re.sub(clean_pattern, '', content, flags=re.DOTALL | re.MULTILINE)
        cleaned_statements = [statement.strip() for statement in re.split(split_pattern, cleaned_content) if statement.strip()]
        cleaned_content = "\n".join(cleaned_statements)

        true_log_positions = [
            i + 1 for i, label in enumerate(labels) if label == 1
        ]
        total_log_lines += len(true_log_positions)

        temp_file_path = "temp.java"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(cleaned_content)

        # Predict log positions
        predictions_df = predict_log_positions(temp_file_path, model, tokenizer, device)
        predicted_positions = predictions_df["LineNumber"].tolist()
        predicted_positions_all.extend(predicted_positions)
        true_log_positions_all.extend(true_log_positions)
        correct_predictions += len(set(true_log_positions) & set(predicted_positions))

  
    accuracy = (correct_predictions / total_log_lines) if total_log_lines > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%     Correct: {correct_predictions}     Total: {total_log_lines}")
    calculate_distance_matrix(predicted_positions_all, true_log_positions_all)

    return accuracy

def calculate_distance_matrix(predicted_positions, true_positions):
    """
    Calculate distance metrics to evaluate how close the predicted log positions are to the true log positions.
    """
    
    predicted_positions.sort()
    true_positions.sort()

    distance_matrix = []

    for true_pos in true_positions:
        row = []
        for pred_pos in predicted_positions:
            row.append(abs(true_pos - pred_pos))
        distance_matrix.append(row)

    
    threshold = 1 
    correct_within_threshold = sum(
        1 for true_pos in true_positions if any(abs(true_pos - pred_pos) <= threshold for pred_pos in predicted_positions)
    )

# csv_file = "./dataset/validation/train_1_sqshq_PiggyMetrics.csv" 

# accuracy = evaluate_model_accuracy_with_distance(csv_file, model, tokenizer, device)

# print(f"Final Model Accuracy: {accuracy  * 100:.2f}%")

# print(predict_log_positions("/home/yhasabe/GENAI4SE_Project/code/dama.java", model, tokenizer, device))
