import numpy as np
import python3.Ch02.kNN as knn
import os


def img2vector(filename):
    """
    read vector from .txt file in digits
    :param filename: .txt filepath
    :return: [1, 1024] vector
    """
    return_vect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_str[j])

    return return_vect


def handwriting_classtest():
    hw_labels = []
    training_file_list = os.listdir('digits/trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        hw_labels.append(class_num)
        training_mat[i, :] = img2vector('digits/trainingDigits/{}'.format(filename_str))

    test_file_list = os.listdir('digits/testDigits')
    error_count = 0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vector_undertest = img2vector('digits/testDigits/{}'.format(filename_str))
        classify_result = knn.classify0(vector_undertest, training_mat, hw_labels, 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classify_result, class_num))
        if classify_result != class_num:
            error_count += 1

    print("the total number of errors is: {}".format(error_count))
    print("the total error rate is: {:.2%}".format(error_count / float(m_test)))


if __name__ == '__main__':
    # test_vector = img2vector('digits/testDigits/0_13.txt')
    # print(test_vector[0, 0:31])
    handwriting_classtest()
