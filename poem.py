import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config

def load_poems(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        poems = file.read().split('\n')
    return [poem.strip() for poem in poems if poem.strip()]

def setup_dataset(poems, tokenizer, max_length=128):
    encoded_poems = [tokenizer.encode(poem, max_length=max_length, truncation=True, return_tensors='tf') for poem in poems]
    padded_poems = tf.keras.preprocessing.sequence.pad_sequences([tf.squeeze(p).numpy() for p in encoded_poems], padding='post')
    dataset = tf.data.Dataset.from_tensor_slices(padded_poems)
    dataset = dataset.map(lambda x: {'input_ids': x, 'labels': x})
    dataset = dataset.shuffle(100).batch(2)
    return dataset

def fine_tune_model(dataset):
    config = GPT2Config.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2', config=config)
  

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss)
    model.fit(dataset, epochs=3)
    return model

def generate_text(model, tokenizer, prompt, length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    generated_text_samples = model.generate(
        input_ids,
        max_length=length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    poems = load_poems('poem.txt')
    dataset = setup_dataset(poems, tokenizer)
    model = fine_tune_model(dataset)

    # Generate text
    prompt = "Under the moonlight"
    print(generate_text(model, tokenizer, prompt))

if __name__ == "__main__":
    main()
