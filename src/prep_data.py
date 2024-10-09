import json
import argparse
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="POS tag words from a JSON file.")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file path.")
    parser.add_argument("-o", "--output", required=True, help="Output file path.")
    args = parser.parse_args()

    # Read words from the input JSON file
    with open(args.input, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Extract words (keys of the JSON object)
    words = list(data.keys())

    # Ensure NLTK resources are downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")

    # Tokenize and POS tag the words
    tagged_words = pos_tag(words)

    # Create a dictionary of words and their POS tags
    tagged_dict = {word: tag for word, tag in tagged_words}

    # Write the tagged words to the output file in JSON format
    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump(tagged_dict, outfile, ensure_ascii=False, indent=2)

    print(f"POS tagging complete. Output written to '{args.output}'.")


if __name__ == "__main__":
    main()
