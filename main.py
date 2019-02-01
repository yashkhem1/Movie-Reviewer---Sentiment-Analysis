import numpy as np
from utils import removePunc, vectorizeReviews, pad_features, tokenize_review, predict
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from model import SentimentRNN
from train import train
from test import test


if __name__ == "__main__":

    # read data from text files
    with open('data/reviews.txt', 'r') as f:
        reviews = f.read()
    with open('data/labels.txt', 'r') as f:
        labels = f.read()

    reviews_split, words = removePunc(reviews)

    words[:30]

    reviews_ints, vocab_to_int = vectorizeReviews(reviews_split, words)

    print('Unique words: ', len((vocab_to_int)))  # should ~ 74000

    print('Tokenized review: \n', reviews_ints[:1])

    # 1=positive, 0=negative label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Zero-length reviews: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    # remove any reviews/labels with zero length from the reviews_ints list.

    # get indices of any reviews with length 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

    seq_length = 200

    features = pad_features(reviews_ints, seq_length=seq_length)

    ## test statements - do not change - ##
    assert len(features) == len(reviews_ints), "Your features should have as many rows as reviews."
    assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

    # print first 10 values of the first 30 batches
    print(features[:30, :10])

    # ## Training, Validation, Test
    #
    # With our data in nice shape, we'll split it into training, validation, and test sets.

    split_frac = 0.8

    # split data into training, validation, and test data (features and labels, x and y)

    split_idx = int(len(features) * 0.8)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    # print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    batch_size = 50

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    print(net)

    # ## Training

    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    net = train(lr, criterion, optimizer, epochs, counter, print_every, clip, train_loader, valid_loader, net, batch_size)

    test(net, test_loader, batch_size, criterion)

    # # Testing by User

    # test_review = input("Enter Test Review")

    # test_ints = tokenize_review(test_review, vocab_to_int)

    # seq_length = 200
    # features = pad_features(test_ints, seq_length)

    # feature_tensor = torch.from_numpy(features)

    # predict(net, test_review, seq_length)
