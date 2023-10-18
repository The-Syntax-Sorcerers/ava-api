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

from gensim.models import Word2Vec
from db import DB
import string


nltk.data.path.append("tmp/custom_nltk_data")
nltk.download(["punkt", "stopwords", "wordnet"], download_dir="tmp/custom_nltk_data")


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


def convert_text_to_vector(file_text, model, vector_size):
    """
    Convert a list of texts into their corresponding word2vec vectors
    """

    vectors = []
    for sentence in file_text:
        words = preprocess_text(sentence)
        vector = np.sum([model.wv[word] for word in words if word in model.wv], axis=0)
        word_count = np.sum([word in model.wv for word in words])
        if word_count != 0:
            vector /= word_count
        else:
            vector = np.zeros(vector_size)
        vectors.append(vector)

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

    payload = {
        'punc_periods': int(punctuations_count['.']),
        'punc_commas': int(punctuations_count[',']),
        'punc_semicolons': int(punctuations_count[';']),
        'punc_colons': int(punctuations_count[':']),
        'punc_exclamations': int(punctuations_count['!']),
        'punc_questions': int(punctuations_count['?']),
        'punc_dashes': int(punctuations_count['-']),
        'punc_open_par': int(punctuations_count['(']),
        'punc_close_par': int(punctuations_count[')']),
        'punc_double_quotes': int(punctuations_count['\"']),
        'punc_apostrophes': int(punctuations_count['\'']),
        'punc_tilda': int(punctuations_count['`']),
        'punc_forward_slash': int(punctuations_count['/']),
    }

    # Return list of punctuation counts
    return payload, list(punctuations_count.values())


def analyze_sentence_lengths(sentences):
    """
  Analyze the lengths of sentences
  """
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    average_length = np.mean(sentence_lengths)
    count_over_avg = np.sum([length > average_length for length in sentence_lengths])
    count_under_avg = np.sum([length < average_length for length in sentence_lengths])
    count_avg = len(sentence_lengths) - count_over_avg - count_under_avg

    payload = {
        'sent_over_avg': float(count_over_avg),
        'sent_under_avg': float(count_under_avg),
        'sent_count_avg': float(count_avg),
        'sent_avg_length': float(average_length),
    }

    return payload, [count_over_avg, count_under_avg, count_avg, average_length]


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

    payload = {
        'word_rare_count': int(rare_count),
        'word_long_count': int(long_count),
        'word_over_avg': float(count_over_avg),
        'word_under_avg': float(count_under_avg),
        'word_count_avg': float(count_avg),
        'word_ttr': float(ttr),
        'word_avg_length': float(average_length)
    }

    return payload, [rare_count, long_count, count_over_avg, count_under_avg, count_avg, ttr]


def calculate_style_vector(texts):
    """
  Calculate the style vector of the texts
  """
    payload1, punctuation_vec = count_punctuations(texts)  # Punctuations stylistic features
    payload2, sentence_vec = analyze_sentence_lengths(texts)  # Sentences stylistic features
    payload3, word_vec = analyze_words(texts)  # Words stylistic features
    word_count = np.sum([len(text.split()) for text in texts])

    vector = np.concatenate((punctuation_vec, sentence_vec, word_vec))

    final_payload = {}
    final_payload.update(payload1)
    final_payload.update(payload2)
    final_payload.update(payload3)
    final_payload.update({'word_count': int(word_count)})

    # DB.store_style_vector(user, filename, final_payload)

    return final_payload, vector / word_count if word_count else vector


def get_vectors(user, past_assign_filenames, w2v_model, is_past_assignment, vector_size):
    res = []
    print("pastnames", past_assign_filenames)
    for filename in past_assign_filenames:
        # read the file from db using filename
        if filename[-4:] == '.npy':
            text_vector_cached = DB.read_past_file(user, filename)
            print("Found Cached Vector, Reading from db", filename)
            res.append(text_vector_cached)
            continue
        
        if is_past_assignment:
            text = DB.read_past_file(user, filename)
            print("past text", text)
        else:
            filename = filename.split('/')[-1]
            text = DB.read_current_assignment(user, filename)
            print("current text", text)


        # Compute text vectors
        w2v_vec = np.mean(convert_text_to_vector(text, w2v_model, vector_size), axis=0)
        print("Current2", text)
        print(type(text))
        final_payload, style_vec = calculate_style_vector(text)

        final_file_vector = np.concatenate((w2v_vec, style_vec), axis=None)
        res.append(final_file_vector)

        # Caching the vector to db only if it is a past assignment (known text)
        if is_past_assignment:

            print("Cached not found, Computing vector", filename)
            # vector has been calculated for a text, so we will cache it in DB
            cached_filename = filename.replace('.txt', '_cached.npy')
            if cached_filename[-4:] != '.npy':
                cached_filename += '_cached.npy'

            # Create an in-memory file
            buffer = io.BytesIO()
            np.save(buffer, final_file_vector, allow_pickle=True)

            # Getting file_data & uploading to DB
            file_data = buffer.getvalue()
            DB.upload_cached_file(user, cached_filename, file_data)
            DB.store_past_style_vector(user, filename, final_payload)

            print("Vector Cached to database", cached_filename)
            buffer.close()

        else:
            return final_payload, res

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
    final_payloads = []
    for key, val in tqdm(corpus.items(), total=len(corpus)):
        print()
        if len(val['unknown']) == 0:
            continue

        final_payload, unknown_vecs = get_vectors(user, val['unknown'], w2v_model, False, vector_size)
        final_payloads.append(final_payload)
        res[key] = {
            'known': get_vectors(user, val['known'], w2v_model, True, vector_size),
            'unknown': unknown_vecs
        }

    return final_payloads, res


def preprocess_dataset(user, current_environment, vector_size=300):
    word2vec_model = Word2Vec.load(current_environment + "w2v_model/word2vec.model")
    final_payloads, test_data = build_corpus_and_vectorize_text_data(user, word2vec_model, vector_size)
    return final_payloads, test_data
