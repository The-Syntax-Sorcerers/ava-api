import io
import os
import json
import nltk
from tqdm import tqdm
import glob
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from db import DB

nltk.download(["punkt", "stopwords", "wordnet"])
from gensim.models import Word2Vec
import string


def gather_corpus_filenames(user):
    unknown_filenames = [DB.construct_path_current(user.subject, user.assignment, user.id)]
    known_filenames = DB.list_past_assignments(user.email)

    new_known_filenames = []
    for filename in known_filenames:
        # get the filename without last 4 letters
        name = filename[:-4]
        if 'cached' in name:
            continue
        else:
            name = name.split('_')[0]
            if name + '_cached.npy' in known_filenames:
                new_known_filenames.append(name + '_cached.npy')
            else:
                new_known_filenames.append(filename)

    # unknown_text = []
    # unknown_file = DB.download_current_assignment(user.subject, user.assignment, user.id)
    # text_lines = []
    # for line in unknown_file.decode('utf-8').splitlines():
    #     cleaned_line = line.strip().lstrip("\ufeff")
    #     text_lines.append(cleaned_line)
    # unknown_text.append(text_lines)

    # known_files = DB.get_past_files(user.email)
    # for known_file in known_files:
    #     text_lines = []
    #     for line in known_file.decode('utf-8').splitlines():
    #         cleaned_line = line.strip().lstrip("\ufeff")
    #         text_lines.append(cleaned_line)
    #     known_text.append(text_lines)

    return new_known_filenames, unknown_filenames


# def build_corpus(user):
#     corpus = {}
#
#     known_text, unknown_text = extract_text_from_files(user)
#     corpus[0] = {
#         'known': known_text,
#         'unknown': unknown_text
#     }
#
#     return corpus


def preprocess_text(text):
    """
    Preprocess a given text by tokenizing, removing punctuation and numbers,
    removing stop words, and lemmatizing.

    Args:
        text (str): The text to preprocess.

    Returns:
        list: The preprocessed text as a list of tokens.
    """
    if not isinstance(text, str):
        text = str(text)

    # Tokenize the text into words
    tokens = word_tokenize(text.lower())

    # Remove punctuation and numbers
    table = str.maketrans('', '', string.punctuation + string.digits)
    tokens = [word.translate(table) for word in tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if (not word in stop_words) and (word != '')]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def convert_text_to_vector(user, past_filename, model, is_past_assignment, vector_size):
    """
    Convert a list of texts into their corresponding word2vec vectors
    """

    if past_filename[-4:] == '.npy':
        pose = DB.read_past_file(user, past_filename)
        print("Found Cached Vector, Reading from db", past_filename)
        return pose

    # We have the filename, bt now loading the file from db
    text = DB.read_past_file(user, past_filename) if is_past_assignment else DB.read_current_assignment(user)

    vectors = []
    for sentence in text:
        words = preprocess_text(sentence)
        vector = np.sum([model.wv[word] for word in words if word in model.wv], axis=0)
        word_count = np.sum([word in model.wv for word in words])
        if word_count != 0:
            vector /= word_count
        else:
            vector = np.zeros(vector_size)
        vectors.append(vector)

    # if it is an unknown text, we don't need to cache it.
    if not is_past_assignment:
        return vectors

    print("Cached not found, Computing vector", past_filename)
    # vector has been calculated for a text, so we will cache it in DB
    cached_filename = past_filename.replace('.txt', '_cached.npy')

    # Create an in-memory file
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=True)

    # Getting file_data & uploading to DB
    file_data = buffer.getvalue()
    DB.upload_cached_file(user, cached_filename, file_data)

    print("Vector Cached to database", cached_filename)
    buffer.close()
    return vectors


def count_punctuations(texts):
    """
  Count the frequency of different punctuations in the texts
  """
    # Define punctuations to count
    punctuations = {'.', ',', ';', ':', '!', '?', '-', '(', ')', '\"', '\'', '`', '/'}

    # Initialize dictionary to count punctuations
    punctuations_count = {p: 0 for p in punctuations}

    # Count punctuations in text_list
    for text in texts:
        for char in text:
            if char in punctuations:
                punctuations_count[char] += 1

    # Return list of punctuation counts
    return list(punctuations_count.values())


def analyze_sentence_lengths(sentences):
    """
  Analyze the lengths of sentences
  """
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    average_length = np.mean(sentence_lengths)
    count_over_avg = np.sum([length > average_length for length in sentence_lengths])
    count_under_avg = np.sum([length < average_length for length in sentence_lengths])
    count_avg = len(sentence_lengths) - count_over_avg - count_under_avg

    return [count_over_avg, count_under_avg, count_avg, average_length]


def analyze_words(texts):
    """
    Analyze the words used in the texts
    """
    words = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for text in texts:
        tokenized = word_tokenize(text.lower())
        processed = [lemmatizer.lemmatize(word) for word in tokenized if word not in stop_words]
        words += processed
    word_freq = nltk.FreqDist(words)
    rare_count = np.sum([freq <= 2 for word, freq in word_freq.items()])
    long_count = np.sum([len(word) > 6 for word in words])
    word_lengths = [len(word) for word in words]
    average_length = np.mean(word_lengths)
    count_over_avg = np.sum([length > average_length for length in word_lengths])
    count_under_avg = np.sum([length < average_length for length in word_lengths])
    count_avg = len(word_lengths) - count_over_avg - count_under_avg
    ttr = len(set(words)) / len(words) if words else 0

    return [rare_count, long_count, count_over_avg, count_under_avg, count_avg, ttr]


def calculate_style_vector(texts):
    """
  Calculate the style vector of the texts
  """
    punctuation_vec = count_punctuations(texts)  # Punctuations stylistic features
    sentence_vec = analyze_sentence_lengths(texts)  # Sentences stylistic features
    word_vec = analyze_words(texts)  # Words stylistic features
    word_count = np.sum([len(text.split()) for text in texts])

    vector = np.concatenate((punctuation_vec, sentence_vec, word_vec))

    return vector / word_count if word_count else vector


def get_vectors(user, past_assign_filenames, w2v_model, is_past_assignment, vector_size):
    res = []
    for filename in past_assign_filenames:
        w2v_vec = np.mean(convert_text_to_vector(user, filename, w2v_model, is_past_assignment, vector_size), axis=0)
        style_vec = calculate_style_vector(filename)
        res.append(np.concatenate((w2v_vec, style_vec), axis=None))

    return res


def build_corpus_and_vectorize_text_data(user, w2v_model, vector_size):
    """
  Build author data from the corpus
  """

    corpus = {}

    known_filenames, unknown_filenames = gather_corpus_filenames(user)
    corpus[0] = {
        'known': known_filenames,
        'unknown': unknown_filenames
    }

    res = {}
    for key, val in tqdm(corpus.items(), total=len(corpus)):
        print()
        if len(val['unknown']) == 0:
            continue
        res[key] = {
            'known': get_vectors(user, val['known'], w2v_model, True, vector_size),
            'unknown': get_vectors(user, val['unknown'], w2v_model, False, vector_size),
        }

    return res


def preprocess_dataset(user, vector_size=300):
    word2vec_model = Word2Vec.load("w2v_model/word2vec.model")
    test_data = build_corpus_and_vectorize_text_data(user, word2vec_model, vector_size)
    return test_data
