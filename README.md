# BERT Sentiment Classification

This repository contains a Jupyter Notebook that implements a sentiment classification model using BERT with PyTorch.

## Features
- Uses **BERT** from the Hugging Face `transformers` library.
- Sentiment classification on the **IMDb dataset**.
- Implements **PyTorch DataLoader** for efficient training.
- Trains with **AdamW optimizer** and evaluates performance using **accuracy, classification report, and confusion matrix**.

## Setup Instructions

### Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch transformers datasets scikit-learn
```

### Dataset
The notebook automatically loads the **IMDb dataset** using:
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

## Model Details
- **Base Model**: `google-bert/bert-base-uncased`
- **Input Size**: `128` tokens max
- **Batch Size**: `16`
- **Epochs**: `3`
- **Optimizer**: AdamW (`lr=2e-5`)

### Model Architecture
- Pretrained **BERT encoder**
- Fully connected classifier:
  - `Linear(768 → 256) → ReLU → Linear(256 → 2)`

## Running the Notebook
Run the Jupyter Notebook step by step to:
1. Load the dataset and tokenizer.
2. Define the **BERT classification model**.
3. Train the model.
4. Evaluate performance.

Or you can export the notebook to **Kaggle Notebook** and start. It is recommended to use a **T100 GPU** for optimal performance.

## Evaluation Metrics
- **Accuracy**
- **Classification Report**
- **Confusion Matrix**

## Results
### BERT Model Training Results
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1/3   | 0.8856              |
| 2/3   | 0.8892              |
| 3/3   | 0.8750              |

**Final Evaluation Metrics**
```
Accuracy: 0.8750

Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.81      0.87     12500
           1       0.83      0.94      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.87     25000
weighted avg       0.88      0.88      0.87     25000

Confusion Matrix:
[[10152  2348]
 [  777 11723]]
```

### GRU Model Training Results
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1/3   | 0.7942              |
| 2/3   | 0.8278              |
| 3/3   | 0.8216              |

**Final Evaluation Metrics**
```
Accuracy: 0.8216

Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.80      0.82     12500
           1       0.81      0.84      0.82     12500

    accuracy                           0.82     25000
   macro avg       0.82      0.82      0.82     25000
weighted avg       0.82      0.82      0.82     25000

Confusion Matrix:
[[10152  2348]
 [  777 11723]]
```

### GRU Model Training Results
| Model |Accuracy             |
|-------|---------------------|
| BERT  | 0.8750              |
| GRU   | 0.8216              |
| Accuracy Difference: | 0.0534 |

## Usage
This model can be extended for various text classification tasks by fine-tuning on different datasets.

---
**Author:** Your Name

