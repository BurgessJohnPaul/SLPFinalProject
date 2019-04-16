import numpy as np
import os
import collections
import random

# Directory where data is stored.
DATA_DIR = "data"

# Number of most frequent words to consider.
NUM_WORDS = 500

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

class NaiveBayes:
    def __init__(self):
        self.name = "Naive Bayes"

        # NOTE: These pos/neg counts refer to truthful and deceptive. A lil misleading, sorry
        self.pos_counts = [0] * NUM_WORDS
        self.neg_counts = [0] * NUM_WORDS
        self.word_to_ind = {}
        self.train_reviews = []

    def train(self, train_reviews, word_to_ind):
        self.word_to_ind = word_to_ind
        self.train_reviews = train_reviews
        num_examples = len(self.train_reviews)

        x_matrix = np.zeros((num_examples, NUM_WORDS))
        y_matrix = np.zeros((num_examples, 1))

        for ind, review in enumerate(self.train_reviews):
            for word in review[0]:
                if word in self.word_to_ind:
                    x_matrix[ind][self.word_to_ind[word]] = 1
                if review[1]:
                    y_matrix[ind] = 1


        for review_truth in self.train_reviews:
            comment = review_truth[0]
            truthfulness = review_truth[1]

            for word in comment:
                if word in self.word_to_ind:
                    if truthfulness:
                        self.pos_counts[self.word_to_ind[word]] += 1
                    else:
                        self.neg_counts[self.word_to_ind[word]] += 1

    def predict(self, review):
        neg_prob = 0
        pos_prob = 0
        for word in review:
            if word in self.word_to_ind:
                num_reviews_split = 0.5 * len(train_reviews)

                pos_prob_pre = self.pos_counts[self.word_to_ind[word]]/float(num_reviews_split)
                neg_prob_pre = self.neg_counts[self.word_to_ind[word]]/float(num_reviews_split)

                if (pos_prob_pre != 0): # avoid log 0 error
                    pos_prob += np.log(pos_prob_pre)
                if (neg_prob_pre != 0):
                    neg_prob += np.log(neg_prob_pre)

        return pos_prob > neg_prob

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
    print ("Accuracy: ", correct / float(len(test_reviews)))

NaiveBayesModel = NaiveBayes()
NaiveBayesModel.train(train_reviews, word_to_ind)
evaluate(NaiveBayesModel, test_reviews)


#big_oof = ["hello", "the", "hotel", "was", "very", "nice", "but", "smelly"]
#print (predict_truth(big_oof, pos_counts, neg_counts, word_to_ind))

'''
np.set_printoptions(threshold=np.inf)
file = open("matrixTest.txt", "+w")
file.write(str(x_matrix))
file.write(str(y_matrix))
file.close()

for review in comment_truth:
    print (review[0])
    print (review[1])
'''
