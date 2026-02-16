"""
Data loading and preprocessing module.

Handles downloading Goodreads reviews from UCSD Book Graph,
sampling, and splitting into train/test sets.
"""

import gzip
import json
import random
import requests


# Genre-to-URL mapping for UCSD Goodreads review data
GENRE_URL_DICT = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(url, head=10000, sample_size=2000):
    """
    Stream reviews from a gzipped JSON URL and return a random sample.

    Args:
        url: URL to a gzipped JSON file with one review per line.
        head: Maximum number of reviews to read from the stream.
        sample_size: Number of reviews to randomly sample from the loaded set.

    Returns:
        List of review text strings.
    """
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    print(f"  Response status: {response.status_code}")

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(head=10000, sample_size=2000):
    """
    Load reviews for all genres from the UCSD Goodreads dataset.

    Args:
        head: Max reviews to stream per genre.
        sample_size: Number of reviews to sample per genre.

    Returns:
        Dictionary mapping genre name to list of review texts.
    """
    genre_reviews = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=head, sample_size=sample_size)
    return genre_reviews


def split_data(genre_reviews_dict, sample_per_genre=1000, train_ratio=0.8):
    """
    Split genre reviews into train and test sets.

    Args:
        genre_reviews_dict: Dict of genre -> list of review texts.
        sample_per_genre: Number of reviews to use per genre.
        train_ratio: Fraction of samples to use for training.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels).
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    train_count = int(sample_per_genre * train_ratio)

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(sample_per_genre, len(reviews)))

        for review in sampled[:train_count]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_count:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")
    return train_texts, train_labels, test_texts, test_labels


if __name__ == "__main__":
    # Quick test: load and split data
    genre_reviews = load_all_genres(head=10000, sample_size=2000)
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    print(f"Genres: {set(train_labels)}")
    print(f"Sample review: {train_texts[0][:100]}...")
