import numpy as np
import os
import collections

# Directory where data is stored.
DATA_DIR = "data"

# Number of most frequent words to consider.
NUM_WORDS = 500

def predict_truth(review, pos_counts, neg_counts, word_to_ind):
    neg_prob = 0
    pos_prob = 0
    for word in review:
        if word in word_to_ind:
            pos_prob += np.log(pos_counts[word_to_ind[word]]/float(800))
            neg_prob += np.log(neg_counts[word_to_ind[word]]/float(800))

    return pos_prob > neg_prob

def populate_vectors(comment_truth, word_list):
    num_examples = 0
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

                num_examples += 1

                for word in first_line:
                    word_list.append(word.lower())
    return num_examples

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
word_to_ind = get_word_to_ind(top_words)

x_matrix = np.zeros((num_examples, NUM_WORDS))
y_matrix = np.zeros((num_examples, 1))

for ind, review in enumerate(comment_truth):
    for word in review[0]:
        if word in word_to_ind:
            x_matrix[ind][word_to_ind[word]] = 1
        if review[1]:
            y_matrix[ind] = 1

pos_counts = [0] * NUM_WORDS
neg_counts = [0] * NUM_WORDS

for review_truth in comment_truth:
    comment = review_truth[0]
    truthfulness = review_truth[1]

    for word in comment:
        if word in word_to_ind:
            if truthfulness:
                pos_counts[word_to_ind[word]] += 1
            else:
                neg_counts[word_to_ind[word]] += 1

correct = 0
predictions = [0, 0]
for review_truth in comment_truth:
    prediction = predict_truth(review_truth[0], pos_counts, neg_counts, word_to_ind)
    predictions[0 if prediction else 1] += 1
    if review_truth[1] == prediction:
        correct += 1

print ("Predictions: ", predictions)
print ("Correct: ", correct)
print ("Naive Bayes Accuracy: ", correct / float(1600))

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

#print (examples)
#print(topNWords)
