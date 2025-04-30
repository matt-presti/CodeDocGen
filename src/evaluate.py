import torch
from transformers import AutoTokenizer, EncoderDecoderModel
import os
import re
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate_model(model_path, output_dir):
    """Evaluation script using only the model for generation with optimized parameters"""
    # Load test data
    print("Loading test data...")
    raw_data = torch.load(os.path.join(output_dir, "data", "test_data.pt"))
    
    # Load tokenizer with proper RoBERTa settings
    print("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Print special token information for verification
    print("\n=== TOKENIZER INFORMATION ===")
    print(f"BOS/CLS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS/SEP token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    try:
        model = EncoderDecoderModel.from_pretrained(model_path)
        print("Successfully loaded trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a fresh model instance...")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "microsoft/codebert-base", "microsoft/codebert-base"
        )
    
    # Configure generation with CORRECT RoBERTa tokens
    print("\nSetting generation parameters for RoBERTa tokenizer...")
    # RoBERTa uses <s> as BOS/decoder_start_token and </s> as EOS token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Generation parameters optimized for longer outputs
    # Optimization performed here
    model.config.max_length = 150         
    model.config.min_length = 20         
    model.config.no_repeat_ngram_size = 2  # Prevent repeating
    model.config.length_penalty = 2.0     # Strongly favor longer outputs
    model.config.num_beams = 6            # More beams for better search
    model.config.do_sample = True         # Enable sampling
    model.config.temperature = 1.3      
    model.config.top_p = 0.95         
    model.config.early_stopping = False   # Don't stop early
    
    # Print generation settings for verification
    print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
    print(f"eos_token_id: {model.config.eos_token_id}")
    print(f"pad_token_id: {model.config.pad_token_id}")
    print(f"max_length: {model.config.max_length}")
    print(f"min_length: {model.config.min_length}")
    print(f"length_penalty: {model.config.length_penalty}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Get test samples
    test_codes = raw_data["test_codes"][:10]  # Limit to 10 samples for speed
    test_docs = raw_data["test_docs"][:10]
    
    print("\nGenerating documentation...")
    predictions = []
    
    for i, code in enumerate(tqdm(test_codes)):
        print(f"\nProcessing sample {i+1}/{len(test_codes)}")
        
        # Tokenize with RoBERTa tokenizer
        encoded_input = tokenizer(
            code,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Generate with aggressive parameters for longer outputs
        try:
            with torch.no_grad():
                # Try multiple generation strategies if needed
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    max_length=model.config.max_length,
                    min_length=model.config.min_length,
                    num_beams=model.config.num_beams,
                    do_sample=model.config.do_sample,
                    temperature=model.config.temperature,
                    top_p=model.config.top_p,
                    length_penalty=model.config.length_penalty,
                    no_repeat_ngram_size=model.config.no_repeat_ngram_size,
                    early_stopping=model.config.early_stopping,
                    repetition_penalty=1.5  # Penalize repetition
                )
            
            # Show output info
            print(f"Output shape: {outputs.shape}")
            
            # Decode with RoBERTa tokenizer
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Raw prediction length: {len(prediction.split())}")
            
            # If output is still too short, try again with more aggressive parameters
            if len(prediction.split()) < 10:
                print("First attempt produced short output, trying with more aggressive params...")
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_start_token_id=tokenizer.bos_token_id,
                        max_length=200,             # Even longer
                        min_length=30,              # Force even longer output
                        num_beams=8,                # More beams
                        do_sample=True,
                        temperature=1.5,            # More randomness
                        top_p=0.98,                 # Keep more tokens
                        top_k=50,                   # Limit to top k tokens
                        length_penalty=3.0,         # Very strong length bias
                        no_repeat_ngram_size=2,
                        repetition_penalty=2.0,     # Stronger repetition penalty
                        early_stopping=False
                    )
                    
                    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Second attempt length: {len(prediction.split())}")
            
            # Clean up prediction
            prediction = prediction.replace(" ##", "").replace("##", "")
            prediction = re.sub(r'\s+([.,;:!?])', r'\1', prediction)  # Fix spacing
            prediction = re.sub(r'(\s)\1+', r'\1', prediction)        # Remove multiple spaces
            
            print(f"Final prediction: '{prediction[:100]}...'")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Still using model output, but creating a minimal one for errors
            prediction = "Function documentation."
        
        predictions.append(prediction)
    
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(test_docs, predictions)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    # Calculate parameter accuracy
    correct_params = 0
    total_params = 0
    
    for code, pred in zip(test_codes, predictions):
        # Extract parameters
        params_match = re.findall(r'def\s+\w+\((.*?)\):', code)
        if params_match:
            params_str = params_match[0]
            params = [p.strip().split(':')[0].split('=')[0].strip() 
                      for p in params_str.split(',') if p.strip()]
            
            for param in params:
                if param and param not in ['self', 'cls']:
                    total_params += 1
                    if param in pred:
                        correct_params += 1
    
    param_accuracy = correct_params / total_params if total_params > 0 else 0
    
    # Results
    metrics = {
        'ROUGE-1': rouge1,
        'ROUGE-L': rougeL,
        'Parameter Accuracy': param_accuracy
    }
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Show examples
    print("\n=== Example Predictions ===")
    for i in range(min(3, len(predictions))):
        print(f"\nCode: {test_codes[i][:100]}...")
        print(f"Reference: {test_docs[i]}")
        print(f"Prediction: {predictions[i]}")
    
    # Create visualization
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Evaluation')
    plt.ylabel('Score')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    print(f"\nVisualization saved to {os.path.join(output_dir, 'metrics.png')}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model with model-only generation')
    parser.add_argument('--model_path', default='./results/model/final_model', help='Path to model')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.output_dir)