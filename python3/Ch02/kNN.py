import numpy as np
import operator

""" kNN summary
1. prepare the dataset for classify
2. normalization for dataset
3. first norm the test input, then classify it.
"""


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, K):
    dataset_size = dataset.shape[0]

    # 将inX 沿着行方向和列方向分别repeat  dataset_size次和1次
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    # 两个点的特征差的L2范数, 或者叫欧几里得距离
    distances = sq_distances ** 0.5
    # 对distances的值进行升序排序(越前面越相似)，返回各个位置对应的distance的indice
    sorted_distances_indices = distances.argsort()
    class_count = {}

    # 计算前K个样本的标签出现的个数
    for i in range(K):
        vote_i_label = labels[sorted_distances_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # 对标签个数进行降序排序， python3不再支持iteritems， 改为items
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    with open(filename) as fr:
        number_of_lines = fr.readlines().__len__()
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []

    with open(filename) as fr:
        index = 0
        for line in fr.readlines():
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append(love_dictionary[list_from_line[-1]])
            index += 1

    return return_mat, class_label_vector


def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(dataset.shape)
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_testvecs = int(m * ho_ratio)
    error_count = 0
    for i in range(num_testvecs):
        classifer_reuslt = classify0(norm_mat[i, :], norm_mat[num_testvecs:m, :], dating_labels[num_testvecs:m], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifer_reuslt, dating_labels[i]))
        if classifer_reuslt != dating_labels[i]:
            error_count += 1

    print("the total error rate is: {:f}".format(error_count / float(num_testvecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])


if __name__ == '__main__':
    # # group, labels = create_dataset()
    # # print(classify0([0, 0], group, labels, 3))
    # love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    # dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    # print(dating_data_mat)
    # print(dating_labels[:20])
    #
    # import matplotlib
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax: plt.Axes = fig.add_subplot(111)
    # # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
    # # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
    # #            15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # sca = ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
    #                  15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # ax.set_xlabel('Frequent Flyier Miles Earned Per Year')
    # ax.set_ylabel('Percentage of Time Spent Playing Video Games')
    # plt.show()
    #
    # norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # print('norm_mat', norm_mat)
    # print('ranges', ranges)
    # print('min_vals', min_vals)

    # dating_class_test()
    classify_person()
