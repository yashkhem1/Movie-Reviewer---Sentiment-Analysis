import numpy as np
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn


# Remove punctutations


def removePunc(reviews):

    # get rid of punctuation
    reviews = reviews.lower()  # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)

    # create a list of words
    words = all_text.split()

    return [reviews_split, words]


def vectorizeReviews(reviews_split, words):

    # Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    # use the dict to tokenize each review in reviews_split
    # store the tokenized reviews in reviews_ints
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    return [reviews_ints, vocab_to_int]


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def tokenize_review(test_review, vocab_to_int):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


def predict(net, test_review, vocab_to_int, sequence_length=200):

    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review, vocab_to_int)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)
    train_on_gpu = torch.cuda.is_available()

    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
