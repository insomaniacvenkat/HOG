
import os.path
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import cv2
from sklearn.svm import SVC
import hog


def get_good_train_set(directory="./NICTA/TrainSet/PositiveSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_bad_train_set(directory="./NICTA/TrainSet/NegativeSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_good_test_set(directory="./NICTA/TestSet/PositiveSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_bad_test_set(directory="./NICTA/TestSet/NegativeSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_hog_descriptor(image):
    image = cv2.resize(image, (64, 128))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = hog.gamma_correction(image, gamma_value)
    gradient = hog.compute_gradients(image)
    cell_histograms, _ = hog.compute_weighted_vote(gradient)
    hog_blocks, _ = hog.normalize_blocks(cell_histograms)
    return hog_blocks.ravel()


if __name__ == '__main__':
    gamma_value = 1.0
    good_set = get_good_train_set()
    image_count = len(good_set)
    good_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image_file in good_set:
        test_image = cv2.imread(image_file)
        good_set_hog[image_index] = get_hog_descriptor(test_image)
        image_index += 1
    good_set_tag = np.ones(image_count)

    bad_set = get_bad_train_set()
    image_count = len(bad_set)
    bad_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image_file in bad_set:
        test_image = cv2.imread(image_file)
        bad_set_hog[image_index] = get_hog_descriptor(test_image)
        image_index += 1
    bad_set_tag = np.zeros(image_count)

    good_test_set = get_good_test_set()
    good_test_image_count = len(good_test_set)
    good_test_set_hog = np.empty((good_test_image_count, 3780))
    image_index = 0
    for image_file in good_test_set:
        test_image = cv2.imread(image_file)
        good_test_set_hog[image_index] = get_hog_descriptor(test_image)
        image_index += 1

    bad_test_set = get_bad_test_set()
    bad_test_image_count = len(bad_test_set)
    bad_test_set_hog = np.empty((bad_test_image_count, 3780))
    image_index = 0
    for image_file in bad_test_set:
        test_image = cv2.imread(image_file)
        bad_test_set_hog[image_index] = get_hog_descriptor(test_image)
        image_index += 1

    train_data = np.concatenate((good_set_hog, bad_set_hog))
    tag_data = np.concatenate((good_set_tag, bad_set_tag))
    C = 1.0  # SVM regularization parameter
    lin_svc = SVC(kernel='linear', C=C).fit(train_data, tag_data)
    rbf_svc = SVC(kernel='rbf', C=C).fit(train_data, tag_data)
    poly_svc = SVC(kernel='poly', C=C, degree=2).fit(train_data, tag_data)

    # title for the classifiers
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial kernel']

    for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        good_test_results = clf.predict(good_test_set_hog)
        #print(good_test_results)
        bad_test_results = clf.predict(bad_test_set_hog)
        #print(bad_test_results)

        print("Results for {}".format(titles[i]))
        print("Accuracy for Positive Cases: {}".format(np.sum(good_test_results) / good_test_image_count * 100))
        print("Accuracy for Negative Cases: {}".format(100 - (np.sum(bad_test_results) / bad_test_image_count * 100)))
        del good_test_results, bad_test_results


