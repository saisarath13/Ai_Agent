# Fine-Tune GPT-2 for Custom Question-Answering

This repository demonstrates how to fine-tune the GPT-2 model using a custom dataset for a question-answering application. The pipeline includes preprocessing data, fine-tuning the model, saving it for deployment, and generating clean responses.

## Prerequisites

Ensure you have the following installed before proceeding:

- Python 3.7+
- Google Colab or Jupyter Notebook
- Required Python libraries:
  - `datasets`
  - `transformers`
  - `torch`
  - `pandas`
  - `matplotlib`

Install the dependencies using pip:

```bash
!pip install -r requirements.txt
```

## Steps

### 1. Data Preparation

1. Convert your dataset from Excel to CSV format:

   - Use the script `data_preparation.py` to convert an Excel file to CSV format and prepare the dataset for training.

2. Load the dataset using the `datasets` library:

   - Use the script `data_loading.py` to load and shuffle the dataset.

### 2. Model Setup

1. Load and configure the GPT-2 tokenizer and model:

   - Use the script `model_setup.py` to initialize and configure the GPT-2 model and tokenizer.

2. Preprocess the data for fine-tuning:

   - Preprocessing logic is included in `data_preprocessing.py`.

3. Define a data collator and training arguments:

   - Use the `training_arguments.py` script to set up all configurations for training.

### 3. Fine-Tune the Model

1. Train the model:

   - Run the script `fine_tune.py` to train the GPT-2 model on your dataset.

2. Save the fine-tuned model:

   - The fine-tuned model will be saved in the `models/` directory.

### 4. Deployment

1. Application Setup:

   - Use the `app.py` script to serve the fine-tuned model for question-answering tasks.

2. Containerization:

   - Use the provided `Dockerfile` to build a containerized application:

     ```bash
     docker build -t fine-tune-gpt2-app .
     ```

   - Run the container:

     ```bash
     docker run -p 5000:5000 fine-tune-gpt2-app
     ```

3. API Endpoint:

   - The application exposes an endpoint to send questions and receive responses.

### 5. Additional Scripts

- `.gitignore`: Excludes unnecessary files from version control.
- `requirements.txt`: Lists all dependencies required for the project.

## Directory Structure

```
project_root/
|
|-- models/                  # Directory for storing fine-tuned models
|-- app.py                   # Script to serve the model via API
|-- Dockerfile               # Dockerfile for containerization
|-- requirements.txt         # Python dependencies
|-- .gitignore               # Git ignore file
|-- data_preparation.py      # Script to prepare the dataset
|-- data_loading.py          # Script to load and shuffle dataset
|-- model_setup.py           # Script to set up the model and tokenizer
|-- data_preprocessing.py    # Script for preprocessing data
|-- training_arguments.py    # Script to define training configurations
|-- fine_tune.py             # Script for training the model
|-- save_model.py            # Script to save the fine-tuned model
|-- load_model.py            # Script to load the saved model
|-- generate_response.py     # Script to generate responses
|-- plot_loss.py             # Script to plot training and validation loss
|-- test_model.py            # Script to test the model
```

## Results

- Fine-tuned GPT-2 generates relevant, clean responses to user-provided questions.
- Training and validation losses are visualized to track model performance.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute as needed.
