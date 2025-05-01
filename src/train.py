# Title: train.py
# Author: Matthew Presti, @matt-presti
# Purpose: Process CodeSearch Net dataset, train and configure codeBERT model for
# docstring generation. 

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

class CodeDocDataset(Dataset):
    """Simple dataset for code documentation tasks"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def process_and_train(output_dir, batch_size=8, learning_rate=5e-5, epochs=3, max_samples=None):
    """Combined function for data processing, model initialization, and training"""
    # Create directories
    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/codebert-base", "microsoft/codebert-base"
    )
    
    # Configure the decoder for generation
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 128
    model.config.num_beams = 4
    model.config.early_stopping = True
    
    # Load dataset
    print("Loading CodeSearchNet dataset...")
    dataset = load_dataset("code_search_net", "python")
    
    # Extract source code and docstrings
    codes = []
    docstrings = []
    
    # Use a smaller subset if specified
    sample_data = dataset["train"]
    if max_samples and max_samples < len(sample_data):
        indices = list(range(max_samples))
        sample_data = sample_data.select(indices)
    
    print(f"Processing {len(sample_data)} samples...")
    
    # Extract code and docstrings
    for item in tqdm(sample_data):
        # Get the function code and documentation
        code = item["func_code_string"].strip()
        docstring = item["func_documentation_string"].strip()
        
        # Skip if code or docstring is empty
        if not code or not docstring:
            continue
            
        codes.append(code)
        docstrings.append(docstring)
    
    # Split data
    train_codes, test_codes, train_docs, test_docs = train_test_split(
        codes, docstrings, test_size=0.1, random_state=42)
    
    # Further split test into validation and test
    val_codes, test_codes, val_docs, test_docs = train_test_split(
        test_codes, test_docs, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_codes)}, Validation: {len(val_codes)}, Test: {len(test_codes)}")
    
    # Tokenize inputs (source code)
    train_encodings = tokenizer(train_codes, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_codes, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_codes, truncation=True, padding=True, max_length=512)
    
    # Tokenize outputs (docstrings)
    with tokenizer.as_target_tokenizer():
        train_labels = tokenizer(train_docs, truncation=True, padding=True, max_length=128)["input_ids"]
        val_labels = tokenizer(val_docs, truncation=True, padding=True, max_length=128)["input_ids"]
        test_labels = tokenizer(test_docs, truncation=True, padding=True, max_length=128)["input_ids"]
    
    # Create dataset objects
    train_dataset = CodeDocDataset(train_encodings, train_labels)
    val_dataset = CodeDocDataset(val_encodings, val_labels)
    test_dataset = CodeDocDataset(test_encodings, test_labels)
    
    # Store raw test data for evaluation
    raw_data = {
        "test_codes": test_codes,
        "test_docs": test_docs
    }
    
    # Save raw test data for later evaluation
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    torch.save(raw_data, os.path.join(output_dir, "data", "test_data.pt"))
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    model_path = os.path.join(model_dir, "final_model")
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save test dataset for evaluation
    torch.save(test_dataset, os.path.join(output_dir, "data", "test_dataset.pt"))
    
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and train the code documentation model')
    parser.add_argument('--output_dir', default='./results', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=100, 
                        help='Maximum number of samples to use (None for all)')
    
    args = parser.parse_args()
    
    model_path = process_and_train(
        args.output_dir, 
        args.batch_size, 
        args.learning_rate, 
        args.epochs, 
        args.max_samples
    )
    
    print(f"Training completed. Model saved at {model_path}")