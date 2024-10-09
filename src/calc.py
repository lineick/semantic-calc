import gensim
import gensim.downloader as api
import numpy as np
import re


class SemCalculator:
    def __init__(self, base_vocab, model_name="word2vec-google-news-300"):
        """
        Initialize the SemCalculator with a pre-trained Word2Vec model.

        Parameters:
        - model_name: Name of the pre-trained model to load.
        - base_vocab: Optional dictionary of words to POS tags to use as the vocabulary.
        """
        print(f"Loading the '{model_name}' model...")
        self.model = api.load(model_name)
        print("Model loaded successfully.\n")

        # Use the provided base vocabulary
        self.word_pos_tags = base_vocab
        self.vocab = set(base_vocab.keys())

    def is_valid_word(self, word):
        """
        Check if a word is valid (alphabetic characters only).

        Parameters:
        - word: The word to check.

        Returns:
        - True if the word is valid, False otherwise.
        """
        # Exclude words with non-alphabetic characters
        return re.match("^[A-Za-z]+(-[A-Za-z]+)*$", word) is not None

    def embed_word(self, word):
        """
        Get the embedding vector for a word.

        Parameters:
        - word: The word to embed.

        Returns:
        - A numpy array representing the word vector.
        """
        if word in self.model:
            return self.model[word]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")

    def compute_similarity(self, word1, word2):
        """
        Compute the cosine similarity between two words.

        Parameters:
        - word1: First word.
        - word2: Second word.

        Returns:
        - Cosine similarity between the embeddings of the two words.
        """
        return self.model.similarity(word1, word2)

    def add_vectors(self, word_or_vec1, word_or_vec2):
        """
        Add two vectors or words.

        Parameters:
        - word_or_vec1: Word or vector.
        - word_or_vec2: Word or vector.

        Returns:
        - Sum of the two vectors.
        """
        vec1 = (
            self.embed_word(word_or_vec1)
            if isinstance(word_or_vec1, str)
            else word_or_vec1
        )
        vec2 = (
            self.embed_word(word_or_vec2)
            if isinstance(word_or_vec2, str)
            else word_or_vec2
        )
        return vec1 + vec2

    def subtract_vectors(self, word_or_vec1, word_or_vec2):
        """
        Subtract the vector of the second word/vector from the first.

        Parameters:
        - word_or_vec1: Word or vector.
        - word_or_vec2: Word or vector.

        Returns:
        - Difference of the two vectors.
        """
        vec1 = (
            self.embed_word(word_or_vec1)
            if isinstance(word_or_vec1, str)
            else word_or_vec1
        )
        vec2 = (
            self.embed_word(word_or_vec2)
            if isinstance(word_or_vec2, str)
            else word_or_vec2
        )
        return vec1 - vec2

    def multiply_vectors(self, word_or_vec1, word_or_vec2):
        """
        Multiply two vectors or words element-wise.

        Parameters:
        - word_or_vec1: Word or vector.
        - word_or_vec2: Word or vector.

        Returns:
        - Element-wise multiplication of the two vectors.
        """
        vec1 = (
            self.embed_word(word_or_vec1)
            if isinstance(word_or_vec1, str)
            else word_or_vec1
        )
        vec2 = (
            self.embed_word(word_or_vec2)
            if isinstance(word_or_vec2, str)
            else word_or_vec2
        )
        return vec1 * vec2

    def find_shared_connection(self, words, topn=5, pos_filter=None):
        """
        Find the shared connection between given words.

        Parameters:
        - words: List of words.
        - topn: Number of top similar words to return.
        - pos_filter: List of POS tags to include (e.g., ['NN', 'NNS'] for nouns).

        Returns:
        - A list of tuples (word, similarity) representing the most similar words.
        """
        word_vectors = [self.embed_word(word) for word in words]
        mean_vector = np.mean(word_vectors, axis=0)
        return self.get_most_similar_words(
            mean_vector, topn=topn, pos_filter=pos_filter
        )

    def get_most_similar_words(self, words_or_vectors, topn=5, pos_filter=None):
        """
        Get the most similar words to a given vector or list of words/vectors.

        Parameters:
        - words_or_vectors: Word, list of words, vector, or list of vectors.
        - topn: Number of top similar words to return.
        - pos_filter: List of POS tags to include.

        Returns:
        - A list of tuples (word, similarity) representing the most similar words.

        Raises:
        - ValueError: If any input word is not in the model's vocabulary.
        """
        # Handle input: convert words to vectors
        if isinstance(words_or_vectors, list):
            vectors = []
            for w in words_or_vectors:
                if isinstance(w, str):
                    vec = self.embed_word(
                        w
                    )  # This will raise ValueError if word not in vocabulary
                    vectors.append(vec)
                else:
                    vectors.append(w)
            mean_vector = np.mean(vectors, axis=0)
        else:
            if isinstance(words_or_vectors, str):
                mean_vector = self.embed_word(
                    words_or_vectors
                )  # Raises ValueError if word not in vocabulary
            else:
                mean_vector = words_or_vectors  # Assume it's a vector

        similar_words = self.model.similar_by_vector(
            mean_vector, topn=topn * 5
        )  # Get more words to filter

        # Filter by POS and base vocabulary
        filtered_words = []
        for word, similarity in similar_words:
            if word not in self.vocab:
                continue
            if self.is_valid_word(word) and (
                pos_filter is None or self.word_pos_tags.get(word, "") in pos_filter
            ):
                filtered_words.append((word, similarity))
            if len(filtered_words) >= topn:
                break
        return filtered_words

    def get_most_unrelated_words(
        self, input_words_or_vectors, topn=10, pos_filter=None
    ):
        """
        Get the most unrelated words to the given words or vectors.

        Parameters:
        - input_words_or_vectors: A word, a list of words, a vector, or a list of vectors.
        - topn: Number of top unrelated words to return.
        - pos_filter: List of POS tags to include.

        Returns:
        - A list of tuples (word, average_distance) representing the most unrelated words.

        Raises:
        - ValueError: If any input word is not in the model's vocabulary.
        """
        # Handle input: convert words to vectors
        if isinstance(input_words_or_vectors, list):
            input_vectors = []
            for w in input_words_or_vectors:
                if isinstance(w, str):
                    vec = self.embed_word(
                        w
                    )  # Raises ValueError if word not in vocabulary
                    input_vectors.append(vec)
                else:
                    print(type(w))
                    input_vectors.append(w)
        else:
            if isinstance(input_words_or_vectors, str):
                vec = self.embed_word(
                    input_words_or_vectors
                )  # Raises ValueError if word not in vocabulary
                input_vectors = [vec]
            else:
                input_vectors = [input_words_or_vectors]

        # Precompute norms of input vectors
        input_norms = [np.linalg.norm(vec) for vec in input_vectors]

        # Prepare to filter out unwanted words
        def is_valid_pos(word):
            if pos_filter is None:
                return True
            return self.word_pos_tags.get(word, "") in pos_filter

        # Compute average cosine distance (1 - similarity) for all words in base vocabulary
        distances = []
        for word in self.vocab:
            if not self.is_valid_word(word):
                continue  # Skip invalid words
            if not is_valid_pos(word):
                continue  # Skip words not matching the POS filter
            if word not in self.model:
                continue  # Skip words not in the model's vocabulary
            word_vector = self.model[word]
            word_norm = np.linalg.norm(word_vector)
            # Compute average distance to all input vectors
            total_distance = 0
            for vec, norm in zip(input_vectors, input_norms):
                sim = np.dot(vec, word_vector) / (norm * word_norm)
                distance = 1 - sim  # Cosine distance
                total_distance += distance
            average_distance = total_distance / len(input_vectors)
            distances.append((word, average_distance))
        # Sort by largest average distance
        distances.sort(key=lambda x: x[1], reverse=True)
        return distances[:topn]
