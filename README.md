# CodeDocGen--Exploring Automated Code Documentation Generation
## Corpus--CodeSearchNet , Model--codeBERT


## Project Overview

This project explores developing a system that automatically generates documentation for Python functions using CodeBERT on the CodeSearchNet dataset. Familiarity with Hugging Face's Transformers architecture will be helpful for using the framework. Evaluation uses NLP metrics (ROUGE-1, ROUGE-L) and practical usefulness assessments (parameter identification accuracy).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/matt-presti/CodeDocGen.git
   cd CodeDocGen
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the model

```
python src/train.py --output_dir ./results --batch_size 8 --epochs 3 --learning_rate 5e-5 --max_samples 1000
```

### Hyperparameters

You can adjust the following parameters:
- `--output_dir`: Default is ./results
- `--batch_size`: Batch size for training (default: 2)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_samples`: Maximum number of samples to use for training (default: 100, use None for all)

### Evaluation

```
python src/evaluate_model.py --model_path ./results/model/final_model --output_dir ./results
```

Parameters for `evaluate.py`:
- `--model_path`: Path to the trained model directory (default: './results/model/final_model')
- `--output_dir`: Output directory for results (default: './results')

## Project Structure

- `src/`: Source code
  - `train.py`: Model training pipeline
  - `evaluate.py`: Docstring generation and evaluation pipeline
- `results/`: Output directory
  - `metrics.png`: Visualization of evaluation results
  - `model/`: Location of saved model checkpoints
- `requirements.txt`: Project dependencies


## Results

Running `train.py` generates a **detailed log** about the tokenization process, model configuration, and generation parameters with output  location **./results/model** 

Running `evaluate_model.py` generates:
- **Metrics summary**: ROUGE-1, ROUGE-L, and parameter identification accuracy scores
- **Metrics visualization**: A bar chart saved as `metrics.png` in the output directory
- **Example predictions**: The script outputs several examples comparing:
  - Original code snippets
  - Reference (human-written) documentation
  - Model-generated documentation


The framework allows for flexible model training and optimization based on computational resources and provides an initial framework for evaluating CodeBERT's utility for automated docstring generation. 

### [Project Report](CodeDocGen_finalreport.pdf)


### [Project Video] (https://www.youtube.com/watch?v=Hip28PYFmdI)

