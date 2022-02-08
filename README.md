# Predicts the fourth word in sequence given the preceding trigram

Consider one particular model for examining sequences of words. The task is to predict the fourth word in sequence given the preceding trigram, e.g., trigram: ‘Neural nets are’, fourth word: ‘awesome’. A database of articles were parsed to store sample fourgrams restricted to a vocabulary size of 250 words. The file assign2_data2.h5 contains training samples for input and output (trainx, traind), for validation (valx, vald), and for testing (testx, testd). Using these samples, the following network should be trained via backpropagation.

<img src="https://user-images.githubusercontent.com/40184143/153077632-102ec94a-744b-4a44-9b57-906d8b6f3f15.png" width="700" height="500">

The input layer has 3 neurons corresponding to the trigram entries. An embedding matrix R (250×D) is used to linearly map each single word onto a vector representation of length D. The same embedding matrix is used for each input word in the trigram, without considering the sequence order. The hidden layer uses a sigmoidal activation function on each of P hidden-layer neurons. The output layer predicts a separate response z_i for each of 250 vocabulary words, and the probability of each word is estimated via a soft-max operation.

## Part A

Assume the following parameters: a stochastic gradient descent algorithm, a mini-batch size of 200 samples, a learning rate of η = 0.15, a momentum rate of α = 0.85, a maximum of 50 epochs, and weights and biases initialized as random Gaussian variables of std 0.01. If necessary, adjust these parameters to improve network performance. The algorithm should be stopped based on the cross-entropy error on the validation data. Experiment with different D and P values, (D,P) = (32,256), (16,128), (8,64) and discuss your results.

## Part B

Pick some sample trigrams from the test data, and generate predictions for the fourth word via the trained neural network. Store the the predicted probability for each of the 250 words. For each of 5 sample trigrams, list the top 10 candidates for the fourth word. Are the network predictions sensible?

