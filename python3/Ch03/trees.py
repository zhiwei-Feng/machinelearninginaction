from math import log
import operator

from python3.Ch03 import treePlotter


def calc_shannon_ent(dataset):
    """
    calculate the shannon entropy of dataset
    :param dataset:
    :return: shannon entropy
    """
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        cur_label = feat_vec[-1]
        if cur_label not in label_counts.keys():
            label_counts[cur_label] = 0
        label_counts[cur_label] += 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = label_counts[key] / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    """
    this method run as follow:
    # >>> myDat,labels=trees.createDataSet()
    # >>> myDat
    # [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']] >>> trees.splitDataSet(myDat,0,1)
    # [[1, 'yes'], [1, 'yes'], [0, 'no']]
    # >>> trees.splitDataSet(myDat,0,0)
    # [[1, 'no'], [1, 'no']]
    in summary: this method will split the dataset which {axis}-th column value equal {value}
    :param dataset:
    :param axis:
    :param value:
    :return:
    """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)

    return ret_dataset


def choose_best_fet_to_split(dataset):
    num_feat = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(num_feat):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for v in unique_val:
            sub_dataset = split_dataset(dataset, i, v)
            prob = len(sub_dataset) / len(dataset)
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i

    return best_feat


def majority_cnt(class_list):
    """
    majority vote from result list of class
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    # class_list store the category of each data
    class_list = [example[-1] for example in dataset]
    # if all class are same, then return the class
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # if the feature num equal 1 means no feature. so return the majority class
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    # choose a best feature for splitting dataset, or means root node
    best_feat = choose_best_fet_to_split(dataset)
    # the labels means feature's name
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    # extract the feature value of the entire dataset
    feat_values = [example[best_feat] for example in dataset]
    # make it unique
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]  # note: the best feature name has already not existed in labels
        # recursive build sub tree
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)

    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree)[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]

    return class_label


def store_tree(input_tree, filename):
    """
    pickle store and load must need binary format
    :param input_tree:
    :param filename:
    :return:
    """
    import pickle
    with open(filename, "wb") as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


if __name__ == '__main__':
    my_dat, labels = create_dataset()
    print(labels)
    # print(calc_shannon_ent(my_dat))
    # print(split_dataset(my_dat, 0, 1))
    # print(split_dataset(my_dat, 0, 0))
    # print(choose_best_fet_to_split(my_dat))
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    print(classify(myTree, labels, [1, 0]))
    print(classify(myTree, labels, [1, 1]))

    store_tree(myTree, 'classifierStorage.txt')
    print(grab_tree('classifierStorage.txt'))
