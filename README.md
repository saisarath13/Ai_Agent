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
!pip install datasets transformers accelerate matplotlib
```

## Steps

### 1. Data Preparation

1. Convert your dataset from Excel to CSV format:

   ```python
   import pandas as pd

   # Load the Excel file
   excel_file = '/content/drive/MyDrive/ML PROJECTS/cleaned_full_personal_dataset.xlsx'
   csv_file = '/content/drive/MyDrive/ML PROJECTS/cleaned_full_personal_dataset.csv'

   # Convert to CSV
   data = pd.read_excel(excel_file)
   data.to_csv(csv_file, index=False)

   print("File converted to CSV successfully!")
   ```

2. Load the dataset using the `datasets` library:

   ```python
   from datasets import load_dataset

   dataset = load_dataset('csv', data_files={'train': csv_file})
   shuffled_dataset = dataset['train'].shuffle(seed=42)  # Randomize for better training
   print(shuffled_dataset)
   ```

### 2. Model Setup

1. Load and configure the GPT-2 tokenizer and model:

   ```python
   from transformers import GPT2Tokenizer, GPT2LMHeadModel

   tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
   model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

   tokenizer.pad_token = tokenizer.eos_token
   model.resize_token_embeddings(len(tokenizer))
   ```

2. Preprocess the data for fine-tuning:

   ```python
   def preprocess_data(examples):
       inputs = [
           tokenizer(
               q + " [SEP] " + a,
               padding="max_length",
               truncation=True,
               max_length=512
           ) for q, a in zip(examples['Question'], examples['Answer'])
       ]
       return {
           'input_ids': [inp['input_ids'] for inp in inputs],
           'attention_mask': [inp['attention_mask'] for inp in inputs],
           'labels': [inp['input_ids'] for inp in inputs]
       }

   tokenized_dataset = shuffled_dataset.map(preprocess_data, batched=True)
   split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
   train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
   ```

3. Define a data collator and training arguments:

   ```python
   from transformers import DataCollatorForLanguageModeling, TrainingArguments

   data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       save_strategy="epoch",
       logging_strategy="steps",
       logging_steps=10,
       learning_rate=3e-5,
       per_device_train_batch_size=4,
       per_device_eval_batch_size=8,
       num_train_epochs=6,
       weight_decay=0.01,
       logging_dir="./logs",
       report_to="none"
   )
   ```

### 3. Fine-Tune the Model

1. Initialize and train the model using the `Trainer` API:

   ```python
   from transformers import Trainer

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       data_collator=data_collator,
   )

   train_results = trainer.train()
   ```

2. Save the fine-tuned model:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   model.save_pretrained('/content/drive/MyDrive/fine_tuned_model')
   tokenizer.save_pretrained('/content/drive/MyDrive/fine_tuned_model')
   ```

### 4. Generate Responses

1. Load the fine-tuned model and tokenizer:

   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer

   model_path = '/content/drive/MyDrive/fine_tuned_model'
   model = GPT2LMHeadModel.from_pretrained(model_path)
   tokenizer = GPT2Tokenizer.from_pretrained(model_path)

   print("Model and tokenizer loaded successfully!")
   ```

2. Define a function to generate responses:

   ```python
   def generate_response(question):
       if question.strip().lower() in ["hi", "hi?", "hello", "hey"]:
           return "Hi, Good day!"

       input_text = f"{question} Answer briefly:"
       inputs = tokenizer(
           input_text,
           return_tensors="pt",
           max_length=50,
           truncation=True
       )
       output = model.generate(
           inputs['input_ids'],
           attention_mask=inputs['attention_mask'],
           max_length=70,
           no_repeat_ngram_size=3,
           repetition_penalty=2.0,
           num_beams=5,
           temperature=0.7,
           top_p=0.9,
           early_stopping=True,
           pad_token_id=tokenizer.eos_token_id
       )
       response = tokenizer.decode(output[0], skip_special_tokens=True)
       response = response.replace(question, "").replace("Answer briefly:", "").strip()
       return response.split(".")[0] + "."
   ```

### 5. Plot Training Loss

Extract and plot training and validation loss:

```python
training_loss = []
validation_loss = []

for log in trainer.state.log_history:
    if 'loss' in log:
        training_loss.append(log['loss'])
    if 'eval_loss' in log:
        validation_loss.append(log['eval_loss'])

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(training_loss, label="Training Loss", marker='o')
plt.plot(validation_loss, label="Validation Loss", marker='x')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

### 6. Example Testing

Test the model with example questions:

```python
test_questions = [
    "Hi?",
    "Why are you interested in AI?",
    "What skills do you have in generative AI?",
    "What is your specialization in AI?",
    "Tell me about your certifications."
]

for question in test_questions:
    print(f"Question: {question}")
    print(f"Response: {generate_response(question)}
")
```

## Results

- Fine-tuned GPT-2 generates relevant, clean responses to user-provided questions.
- Training and validation losses are visualized to track model performance.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute as needed.

