from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


output_dir = Path("output")
tokenizer_path = output_dir / "bpe_tokenizer.json"

special_tokens = ["<|endoftext|>"]

# Preparing the text iterator
def get_texts(dataset):
    for item in dataset:
        if item["text"].strip():  # Skipping empty lines
            yield item["text"]

def bpe_tokenizer():

    # Initializing BPE tokenizer
    tokenizer = Tokenizer(BPE())

    # Applying normalization and pre-tokenization
    tokenizer.normalizer = Lowercase()

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer

def train_bpe_tokenizer(tokenizer, dataset):

    trainer = BpeTrainer(
        vocab_size=500,
        min_frequency=2,
        special_tokens=special_tokens
    )

    # Training the tokenizer
    tokenizer.train_from_iterator(
        get_texts(dataset),
        trainer=trainer
    )
    
    return tokenizer

def main():

    # Loading the dataset
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train"
    )

    # Training the BPE tokenizer
    tokenizer = bpe_tokenizer()

    tokenizer = train_bpe_tokenizer(tokenizer, dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(tokenizer_path))

    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    main()