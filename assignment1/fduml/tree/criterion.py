"""
criterion
"""

import math
import numpy as np

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """ Count the number of labels of nodes """
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels

def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    all_labels = np.array(list(all_labels.values()))
    left_labels = np.array(list(left_labels.values()))
    right_labels = np.array(list(right_labels.values()))
    n = np.sum(all_labels)
    entropy_p = 0
    for t in all_labels:
        p_t = t / n
        entropy_p -= p_t * math.log2(p_t)

    n1 = np.sum(left_labels)
    entropy_left = 0
    for t in left_labels:
        p_t = t / n1
        entropy_left -= p_t * math.log2(p_t)

    n2 = np.sum(right_labels)
    entropy_right = 0
    for t in right_labels:
        p_t = t / n2
        entropy_right -= p_t * math.log2(p_t)

    info_gain = entropy_p - n1/(n1+n2) * entropy_left - n2/(n1+n2) * entropy_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    n = np.sum(np.array(list(all_labels.values())))
    n1 = np.sum(np.array(list(left_labels.values())))
    n2 = np.sum(np.array(list(right_labels.values())))
    if n1 == 0 or n2 == 0:
        return 0
    else:
        split_ratio = - n1/n*np.log2(n1/n) - n2/n*np.log2(n2/n)
    info_gain = info_gain / split_ratio

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain



def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    all_labels = np.array(list(all_labels.values()))
    left_labels = np.array(list(left_labels.values()))
    right_labels = np.array(list(right_labels.values()))
    before = 1 - np.sum(np.square(all_labels)) / np.sum(all_labels)**2
    
    if len(left_labels) == 0:
        after_left = 0
    else:
        after_left = 1 - np.sum(np.square(left_labels)) / np.sum(left_labels)**2
    
    if len(right_labels) == 0:
        after_right = 0
    else:
        after_right = 1 - np.sum(np.square(right_labels)) / np.sum(right_labels)**2
    
    after = after_left + after_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """ Calculate the error rate """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    all_labels = np.array(list(all_labels.values()))
    left_labels = np.array(list(left_labels.values()))
    right_labels = np.array(list(right_labels.values()))
    left_all = np.sum(left_labels)
    right_all = np.sum(right_labels)
    if(left_all == 0 or right_all == 0):    return 0
    before = 1 - max(all_labels) / np.sum(all_labels)
    after_left = 1 - max(left_labels) / np.sum(left_labels)
    after_right = 1 - max(right_labels) / np.sum(right_labels)
    n1 = l_y.reshape(-1).shape[0]
    n2 = r_y.reshape(-1).shape[0]
    n = n1 + n2
    after = n1/n * after_left + n2/n * after_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
