import numpy as np
import os
import collections


def predict_truth(review, pos_counts, neg_counts, word_to_ind):
    neg_prob = 0
    pos_prob = 0
    for word in review:
        if word in word_to_ind:
            print(word_to_ind[word])
            pos_prob += np.log(pos_counts[word_to_ind[word]]/float(800))
            neg_prob += np.log(neg_counts[word_to_ind[word]]/float(800))

    return pos_prob > neg_prob



rootdir = "C:\\Users\\burge\\PycharmProjects\\SLPFinal\\op_spam_v1.4"

num_examples = 0
word_list = []
comment_truth = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
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
        #print (first_line.split(" "))

        comment_truth.append((first_line, truthful))

        num_examples += 1
        for word in first_line:
            word_list.append(word.lower())

word_num = 500
topWords_filthy = collections.Counter(word_list).most_common(word_num)

topWords = []
for tuple in topWords_filthy:
    topWords.append(tuple[0])


word_to_ind = {}
for ind, word in enumerate(topWords):
    word_to_ind[word] = ind
print (word_to_ind)

x_matrix = np.zeros((num_examples, word_num))
y_matrix = np.zeros((num_examples, 1))

for ind, review in enumerate(comment_truth):
    for word in review[0]:
        if word in word_to_ind:
            x_matrix[ind][word_to_ind[word]] = 1
        if review[1]:
            y_matrix[ind] = 1

pos_counts = [0] * word_num
neg_counts = [0] * word_num

for review_truth in comment_truth:
    comment = review_truth[0]
    truthfulness = review_truth[1]
    #if not truthfulness:
    #    print (truthfulness)

    for word in comment:
        if word in word_to_ind:
            if truthfulness:
                pos_counts[word_to_ind[word]] += 1
            else:
                neg_counts[word_to_ind[word]] += 1

print (pos_counts)
print (neg_counts)

big_oof = ["hello", "the", "hotel", "was", "very", "nice", "but", "smelly"]

correct = 0
for review_truth in comment_truth:
    prediction = predict_truth(review_truth[0], pos_counts, neg_counts, word_to_ind)
    if not review[1]:
        print(review[1])
    if review[1] == prediction:
        correct += 1
print ("Correct: ", correct)
print ("Naive Bayes Accuracy: ", correct / float(1600))
#print (predict_truth(big_oof, pos_counts, neg_counts, word_to_ind))

'''
np.set_printoptions(threshold=np.inf)
file = open("matrixTest.txt", "+w")
file.write(str(x_matrix))
file.write(str(y_matrix))
file.close()

for review in comment_truth:
    print (review[0])
    print (review[1])'''

#print (examples)
#print(topNWords)