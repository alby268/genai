import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from tensorflow.keras.optimizers import Adam

# Load GPT-2 tokenizer and model (TensorFlow version)
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Ensure the padding token is set to the EOS token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load and tokenize dataset (poem.txt file)
def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    text = " ".join(lines)
    
    # Tokenize the input text into tensors with attention masks
    inputs = tokenizer(text, return_tensors="tf", max_length=block_size, truncation=True, padding='max_length')
    
    # Add attention masks
    inputs["attention_mask"] = tf.cast(inputs["input_ids"] != tokenizer.pad_token_id, dtype=tf.int32)
    
    inputs["labels"] = inputs.input_ids  # Set the labels to input_ids
    return inputs

# Load dataset from poem.txt
dataset = load_dataset("poem.txt", tokenizer)

# Prepare optimizer and loss
optimizer = Adam(learning_rate=5e-5)

# Compile the model with optimizer and loss
model.compile(optimizer=optimizer, loss=model.compute_loss)

# Define training function
def train_model(dataset, epochs=3, batch_size=1):
    steps_per_epoch = len(dataset["input_ids"]) // batch_size
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step in range(steps_per_epoch):
            input_ids = dataset["input_ids"][step * batch_size: (step + 1) * batch_size]
            attention_mask = dataset["attention_mask"][step * batch_size: (step + 1) * batch_size]
            labels = dataset["labels"][step * batch_size: (step + 1) * batch_size]
            
            # Perform training step
            with tf.GradientTape() as tape:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if step % 100 == 0:
                print(f"Step {step}/{steps_per_epoch}, Loss: {loss.numpy()}")

# Train the model
train_model(dataset, epochs=3)

# Save the model after training
model.save_pretrained("./gpt2-poetry-finetuned")
tokenizer.save_pretrained("./gpt2-poetry-finetuned")

# Text generation function based on user prompt
def generate_text_in_poem_world(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    attention_mask = tf.cast(inputs != tokenizer.pad_token_id, dtype=tf.int32)
    
    # Generate text using the model
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, 
                             no_repeat_ngram_size=2, top_p=0.95, temperature=0.8, do_sample=True, 
                             pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage: Generate text based on user input in "poem world"
user_prompt = "In the quiet of the night"
generated_poem = generate_text_in_poem_world(user_prompt)
print(generated_poem)
