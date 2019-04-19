import sys
import numpy as np
import os
import collections
import random
import keras
from keras.layers import Embedding, Dense, LSTM, TimeDistributed
# Directory where data is stored.
DATA_DIR = "data"

# Number of most frequent words to consider.
NUM_WORDS = 1000

#Int val that represents an unknown word
unknown_val = NUM_WORDS
#Int val that represents padding
padding_val = NUM_WORDS + 1

# Prooporttion of dataset in train vs. test set.
# Each class (pos-truthful, neg-deceptive, neg-truthful, neg-deceptive) will be split by this proportion.
# These should not exceed 1
TRAIN_SPLIT = .8
TEST_SPLIT = .2

def populate_vectors(train_reviews, test_reviews, word_list):
    """
    Given three empty vectors, populate based on TRAIN_SPLIT, TEST_SPLIT

    test_reviews [(review, truthfulness, positive), ...]

    """
    for subdir, dirs, files in os.walk(DATA_DIR):
        if len(files) > 10: # Identifying the class folders
            print("Loaded ", subdir)

            rand_files = list(filter(lambda x: ".txt" in x, files))
            random.shuffle(rand_files)

            left_index_train = int(round(TRAIN_SPLIT*len(rand_files)))
            right_index_test = int(round(TEST_SPLIT*len(rand_files))) + left_index_train

            i = 0
            for file in rand_files:

                path_file = os.path.join(subdir, file)
                with open(path_file) as f:
                    first_line = f.readline()
                #print (first_line)
                truthful = "truthful" in path_file

                first_line = first_line.replace(".", " ")
                first_line = first_line.replace(",", " ")
                first_line = first_line.replace("(", " ")
                first_line = first_line.replace(")", " ")
                first_line = first_line.replace("?", "")
                first_line = first_line.replace("!", "")
                first_line = first_line.replace(";", "")

                first_line = first_line.split()

                review_truth = (first_line, truthful)

                if (i < left_index_train):
                    train_reviews.append(review_truth)
                elif (i < right_index_test):
                    test_reviews.append(review_truth)
                else:
                    break

                for word in first_line:
                    word_list.append(word.lower())

                i += 1

def get_top_words(word_list):
    top_words_filthy = collections.Counter(word_list).most_common(NUM_WORDS)
    top_words = []
    for tuple in top_words_filthy:
        top_words.append(tuple[0])
    return top_words

def get_word_to_ind(top_words):
    """
    Given list of words usage with repetitions, return sorted list of most-frequent to least-frequent words (ids).
    """
    word_to_ind = {}
    for ind, word in enumerate(top_words):
        word_to_ind[word] = ind
    return word_to_ind

word_list = []
train_reviews = []
test_reviews = []
num_examples = populate_vectors(train_reviews, test_reviews, word_list)
top_words = get_top_words(word_list)
# print(top_words)
word_to_ind = get_word_to_ind(top_words)

class LSTM_Model:
    def __init__(self):
        self.name = "LSTM Model"

        self.word_to_ind = {}
        self.train_reviews = []

    def train(self, train_reviews, word_to_ind, test_reviews):
        self.word_to_ind = word_to_ind
        self.train_reviews = train_reviews
        num_examples = len(self.train_reviews)

        #Computing 95th percentile sentence length
        review_lens = []
        for review_truth in train_reviews:
            #print (len(review_truth[0]))
            review_lens.append(len(review_truth[0]))
        max_sent_len = int(np.percentile(review_lens, 95))
        print (max_sent_len)

        x_matrix = np.zeros((num_examples, max_sent_len))
        y_matrix = np.zeros((num_examples, 1))

        for example_ind, review_truth in enumerate(self.train_reviews):
            review = review_truth[0]
            truth = review_truth[1]

            for word_index in range(max_sent_len):
                if word_index < len(review):
                    word = review[word_index]
                    if word in self.word_to_ind:
                        word_int = word_to_ind[word]
                    else:
                        word_int = unknown_val
                else:
                    word_int = padding_val
                x_matrix[example_ind][word_index] = word_int

            if truth:
                y_matrix[example_ind] = 1
        #print(x_matrix[0])
        np.set_printoptions(threshold=sys.maxsize)
        file1 = open("x_matrix.txt", "+w")
        file1.write(str(x_matrix))
        file1.close()
        file2 = open("y_matrix.txt", "+w")
        file2.write(str(y_matrix))
        file2.close()

        # KERAS TIME
        EMBED_DIM = 64
        LSTM_DIM = 64
        NUM_EPOCHS = 20

        embedding = Embedding(NUM_WORDS + 2, EMBED_DIM)  # Unknown and padding
        lstm = LSTM(LSTM_DIM, return_sequences=False, input_shape=(max_sent_len, EMBED_DIM))
        dense = Dense(1, activation='sigmoid')

        model = keras.models.Sequential()
        model.add(embedding)
        model.add(lstm)
        model.add(dense)

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        model.fit(x_matrix, y_matrix, epochs=NUM_EPOCHS)

        #Test Data Time
        test_matrix = np.zeros((len(test_reviews), max_sent_len))
        for example_ind, review_truth in enumerate(test_reviews):
            review = review_truth[0]
            truth = review_truth[1]

            for word_index in range(max_sent_len):
                if word_index < len(review):
                    word = review[word_index]
                    if word in self.word_to_ind:
                        word_int = word_to_ind[word]
                    else:
                        word_int = unknown_val
                else:
                    word_int = padding_val
            test_matrix[example_ind][word_index] = word_int
        predicted_y = model.predict(test_matrix)

        correct_count = 0
        for i in range(len(test_reviews)):
            predicted_truth = predicted_y[i][0] > 0.5
            if predicted_truth == test_reviews[i][1]:
                correct_count += 1
        print (correct_count / float (len(test_reviews)))


'''
def evaluate(model, test_reviews):
    correct = 0
    predictions = [0, 0]
    for review_truth in test_reviews:
        prediction = model.predict(review_truth[0])
        predictions[0 if prediction else 1] += 1
        if review_truth[1] == prediction:
            correct += 1

    print ("--- {} ---".format(model.name))
    print ("Split: (train: {}, test: {})".format(TRAIN_SPLIT, TEST_SPLIT))
    print ("Predictions: ", predictions)
    print ("Correct: ", correct)
    print ("Accuracy: ", correct / float(len(test_reviews)))'''

LSTM_Model = LSTM_Model()
LSTM_Model.train(train_reviews, word_to_ind, test_reviews)