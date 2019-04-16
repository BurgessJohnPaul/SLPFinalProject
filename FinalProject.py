import numpy as np
import os
import collections

# Directory where data is stored.
DATA_DIR = "data"

# Number of most frequent words to consider.
NUM_WORDS = 500

def populate_vectors(comment_truth, word_list):
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            path_file = os.path.join(subdir, file)
            if ".txt" in path_file:
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

                comment_truth.append((first_line, truthful))

                for word in first_line:
                    word_list.append(word.lower())

def get_top_words(word_list):
    top_words_filthy = collections.Counter(word_list).most_common(NUM_WORDS)
    top_words = []
    for tuple in top_words_filthy:
        top_words.append(tuple[0])
    return top_words

def get_word_to_ind(top_words):
    '''
    Given list of words usage with repetitions, return sorted list of most-frequent to least-frequent words (ids).
    '''
    word_to_ind = {}
    for ind, word in enumerate(top_words):
        word_to_ind[word] = ind
    return word_to_ind

word_list = []
comment_truth = []
num_examples = populate_vectors(comment_truth, word_list)
top_words = get_top_words(word_list)
# print(top_words)
word_to_ind = get_word_to_ind(top_words)

class NaiveBayes:
    def __init__(self):
        self.pos_counts = [0] * NUM_WORDS
        self.neg_counts = [0] * NUM_WORDS

    def train(self, comment_truth, word_to_ind):
        num_examples = len(comment_truth)

        x_matrix = np.zeros((num_examples, NUM_WORDS))
        y_matrix = np.zeros((num_examples, 1))

        for ind, review in enumerate(comment_truth):
            for word in review[0]:
                if word in word_to_ind:
                    x_matrix[ind][word_to_ind[word]] = 1
                if review[1]:
                    y_matrix[ind] = 1


        for review_truth in comment_truth:
            comment = review_truth[0]
            truthfulness = review_truth[1]

            for word in comment:
                if word in word_to_ind:
                    if truthfulness:
                        self.pos_counts[word_to_ind[word]] += 1
                    else:
                        self.neg_counts[word_to_ind[word]] += 1

    def predict(self, review, word_to_ind):
        neg_prob = 0
        pos_prob = 0
        for word in review:
            if word in word_to_ind:
                pos_prob += np.log(self.pos_counts[word_to_ind[word]]/float(800))
                neg_prob += np.log(self.neg_counts[word_to_ind[word]]/float(800))

        return pos_prob > neg_prob

def evaluate(model):
    correct = 0
    predictions = [0, 0]
    for review_truth in comment_truth:
        prediction = model.predict(review_truth[0], word_to_ind)
        predictions[0 if prediction else 1] += 1
        if review_truth[1] == prediction:
            correct += 1

    print ("Predictions: ", predictions)
    print ("Correct: ", correct)
    print ("Accuracy: ", correct / float(1600))

NaiveBayesModel = NaiveBayes()
NaiveBayesModel.train(comment_truth, word_to_ind)
evaluate(NaiveBayesModel)


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
