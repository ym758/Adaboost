import numpy
from random import randint
import copy

# version 3.9

def get_points_from_file():
    # https://reshetech.co.il/python-tutorials/reading-and-writing-files-in-python
    f = open('four_circle.txt', 'r', encoding='utf-8')
    all_lines = f.readlines()
    f.close()
    points = []
    # https://www.w3schools.com/python/ref_string_replace.asp
    for txt in all_lines:
        piece = txt.replace("\n", "")
        piece = piece.split(' ')
        points.append([float(piece[0]), float(piece[1]), int(piece[2])])
    return points


def divide_points_to_train_test(points):
    train = []
    test = []
    lenght_train = 75
    lenght_test = 75
    for point in points:
        random = randint(0, 1)
        if (len(train) == lenght_train):
            test.append(point)
        elif (len(test) == lenght_test):
            train.append(point)
        elif random == 0:
            train.append(point)
        elif random == 1:
            test.append(point)
    return train, test

def rules(points):
    rules = []
    lenght = len(points)
    for i in range(0, lenght-1):  # 0..74
        for j in range(i+1, lenght):  # 1..75
            rules.append([points[i], points[j], 1])
            rules.append([points[i], points[j], -1])
    return rules

def Classification_of_points(rule, point):
    if rule[0][1] == rule[1][1]:  # points y1, y2 of rule equals
        if rule[2] == 1:
            if point[1] > rule[0][1]:  # point[1] is y point
                return 1  # side right and above is +
            else:
                return -1
        else:
            if point[1] > rule[0][1]:
                return -1
            else:
                return 1

    elif rule[0][0] == rule[1][0]:  # x equals
        if rule[2] == 1:
            if point[0] > rule[0][0]:
                return 1
            else:
                return -1
        else:
            if point[0] > rule[0][0]:
                return -1
            else:
                return 1

    elif rule[0][0] != rule[1][0]:  # x1 rule != x2 rule. y=mx+n
        m = (rule[1][1] - rule[0][1]) / (rule[1][0] - rule[0][0])  # m=(y2-y1) / x2-x1
        n = rule[0][1] - m * rule[0][0]
        y = m * point[0] + n
        if rule[2] == 1:
            if point[1] > y:
                return 1
            else:
                return -1
        else:
            if point[1] > y:
                return -1
            else:
                return 1

def adaboost(rules, points):
    weights_points = {}
    weight_of_rule = {}
    for point in points:
        weights_points.update({str(point): 1 / len(points)})
        weight_of_rule.update({str(point): 0})
    weak_rule_for_iteration = []
    zt = 1
    for i in range(8):
        min_error = numpy.inf
        weak_rule = None
        label_of_rule = None
        for rule in rules:
            error = 0
            for point in points:
                weight_of_rule[str(point)] = point[2]
                label = Classification_of_points(rule, point)
                if label != point[2]:
                    error += (weights_points[str(point)]/zt)
                    weight_of_rule[str(point)] = -1 * point[2]
            if error < min_error and error:
                min_error = error
                weak_rule = rule
                label_of_rule = copy.deepcopy(weight_of_rule)
        alpha = 0.5 * numpy.log((1 - min_error) / min_error)
        if weak_rule:
            weak_rule_for_iteration.append([weak_rule, alpha])
        temp_zt = 0
        for point in points:
            temp_weight = (1 / zt) * weights_points[str(point)] *\
                          (numpy.e ** (-1 * alpha * point[2] * label_of_rule[str(point)]))
            weights_points[str(point)] = temp_weight
            temp_zt += temp_weight
        zt = temp_zt
    return weak_rule_for_iteration

def final_decision_function(rules, points):
    error = 0
    for point in points:
        f_x = 0  # sigma of alpha * label
        for rule in rules:  # rule[1]=alpha
            label = Classification_of_points(rule[0], point)
            f_x += label * rule[1]
        if f_x * point[2] < 0:
            error += 1
    return error / len(points)

def main():

    points = get_points_from_file()
    iterations = 100  # choose number
    k = 8
    error_train = numpy.zeros(k)
    error_test = numpy.zeros(k)
    for i in range(iterations):
        print("iteration number: " + str(i + 1))
        train, test = divide_points_to_train_test(points)
        weak_rules = adaboost(rules(train), train)
        for rules_num in range(k):
            error_train[rules_num] += final_decision_function(weak_rules[:rules_num+1], train)
            error_test[rules_num] += final_decision_function(weak_rules[:rules_num+1], test)
    round = 1
    for train, test in zip(error_train, error_test):
        print("round= " + str(round))
        print("error_train= " + str(train / iterations))
        print("error_test= " + str(test / iterations))
        round += 1

if __name__ == "__main__":
    main()

