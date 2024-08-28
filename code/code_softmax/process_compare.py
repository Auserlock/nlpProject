import matplotlib.pyplot as plt
import numpy as np
from softmax_regression import Softmax


def train_alphas(alpha, total_times, sample, strategy="shuffle", mini_size=None):
    """Train the model with different learning rates."""
    soft = Softmax(len(sample.train), sample.len, 5)
    soft.regression(sample.train_matrix, sample.train_sentiment, alpha, total_times, strategy, mini_size)
    r_train, r_test = soft.correct_rate(sample.train_matrix, sample.train_sentiment, sample.test_matrix,
                                        sample.test_sentiment)
    return r_train, r_test


def alpha_gradient(bag, bigram, trigram, total_times, mini_size):
    """Plot categorization verses different parameters."""
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    # Bag of words
    # Shuffle
    bag_shuffle_train = []
    bag_shuffle_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, total_times, bag, "shuffle")
        bag_shuffle_train.append(r_train)
        bag_shuffle_test.append(r_test)
    # Batch
    bag_batch_train = []
    bag_batch_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / bag.maxitem), bag, "batch")
        bag_batch_train.append(r_train)
        bag_batch_test.append(r_test)
    # Mini-batch
    bag_mini_train = []
    bag_mini_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / mini_size), bag, "mini", mini_size)
        bag_mini_train.append(r_train)
        bag_mini_test.append(r_test)

    # Bigram
    # Shuffle
    bigram_shuffle_train = []
    bigram_shuffle_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, total_times, bigram, "shuffle")
        bigram_shuffle_train.append(r_train)
        bigram_shuffle_test.append(r_test)
    # Batch
    bigram_batch_train = []
    bigram_batch_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / bigram.maxitem), bigram, "batch")
        bigram_batch_train.append(r_train)
        bigram_batch_test.append(r_test)
    # Mini-batch
    bigram_mini_train = []
    bigram_mini_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / mini_size), bigram, "mini", mini_size)
        bigram_mini_train.append(r_train)
        bigram_mini_test.append(r_test)

    # Trigram
    # Shuffle
    trigram_shuffle_train = []
    trigram_shuffle_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, total_times, trigram, "shuffle")
        trigram_shuffle_train.append(r_train)
        trigram_shuffle_test.append(r_test)
    # Batch
    trigram_batch_train = []
    trigram_batch_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / trigram.maxitem), trigram, "batch")
        trigram_batch_train.append(r_train)
        trigram_batch_test.append(r_test)
    # Mini-batch
    trigram_mini_train = []
    trigram_mini_test = []
    for alpha in alphas:
        r_train, r_test = train_alphas(alpha, int(total_times / mini_size), trigram, "mini", mini_size)
        trigram_mini_train.append(r_train)
        trigram_mini_test.append(r_test)

    # Plot
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 2, 1)
    plt.semilogx(alphas, bag_shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, bag_batch_train, 'g--', label='batch')
    plt.semilogx(alphas, bag_mini_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, bag_shuffle_train, 'ro-', alphas, bag_batch_train, 'g+-', alphas, bag_mini_train, 'b^-')
    plt.legend()
    plt.title("Bag of words -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.subplot(3, 2, 2)
    plt.semilogx(alphas, bag_shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, bag_batch_test, 'g--', label='batch')
    plt.semilogx(alphas, bag_mini_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, bag_shuffle_test, 'ro-', alphas, bag_batch_test, 'g+-', alphas, bag_mini_test, 'b^-')
    plt.legend()
    plt.title("Bag of words -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.subplot(3, 2, 3)
    plt.semilogx(alphas, bigram_shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, bigram_batch_train, 'g--', label='batch')
    plt.semilogx(alphas, bigram_mini_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, bigram_shuffle_train, 'ro-', alphas, bigram_batch_train, 'g+-', alphas, bigram_mini_train,
                 'b^-')
    plt.legend()
    plt.title("Bigram -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.subplot(3, 2, 4)
    plt.semilogx(alphas, bigram_shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, bigram_batch_test, 'g--', label='batch')
    plt.semilogx(alphas, bigram_mini_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, bigram_shuffle_test, 'ro-', alphas, bigram_batch_test, 'g+-', alphas, bigram_mini_test, 'b^-')
    plt.legend()
    plt.title("Bigram -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.subplot(3, 2, 5)
    plt.semilogx(alphas, trigram_shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, trigram_batch_train, 'g--', label='batch')
    plt.semilogx(alphas, trigram_mini_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, trigram_shuffle_train, 'ro-', alphas, trigram_batch_train, 'g+-', alphas, trigram_mini_train,
                 'b^-')
    plt.legend()
    plt.title("Trigram -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.subplot(3, 2, 6)
    plt.semilogx(alphas, trigram_shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, trigram_batch_test, 'g--', label='batch')
    plt.semilogx(alphas, trigram_mini_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, trigram_shuffle_test, 'ro-', alphas, trigram_batch_test, 'g+-', alphas, trigram_mini_test,
                 'b^-')
    plt.legend()
    plt.title("Trigram -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.tight_layout()
    plt.show()