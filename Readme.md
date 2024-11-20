# GPT Language Model

This repository contains the implementation of a GPT-based language model using PyTorch. The model is trained on a text dataset and can be used for Shakespeare play generation tasks.

## Repository Structure
- `dataloader.py`: Contains the `DataLoader` class for loading and processing the dataset.
- `dataset/`: Directory containing the text dataset.
- `hyperparameters.py`: Contains the `Hyperparameters` class for managing model hyperparameters.
- `model_2000Itr.pth`: Pre-trained model checkpoint.
- `model.py`: Contains the implementation of the GPT language model and its components.
- `out.txt`: Output file containing generated text samples.
- `run.py`: Script for training, evaluating, and running inference with the model.
- `ss-gpt_dev_notebook.ipynb`: Jupyter notebook for development and experimentation.
- `tokenizer.py`: Contains the `Tokenizer` and `SimpleTokenizer` classes for encoding and decoding text.

## Getting Started

### Prerequisites

- Python 3.12
- PyTorch
- CUDA
- Jupyter Notebook (optional, for running the notebook)

### Installation
1. Install the required packages:
```sh
pip install -r requirements.txt
```

### Usage

#### Training the Model

To train the model, run the following command:
```sh
python run.py train --checkpoint-output model.pth
```

#### Evaluate the model
To evaluate the model, run the following command:
```sh
python run.py eval --checkpoint model_2000Itr.pth --output out.txt --inferece-token-size 3000
```

This repository demonstrates the implementation of a transformer-based language model and provides a foundation for further experimentation and development in natural language processing tasks.