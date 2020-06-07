# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:55:11 2020

@author: EmrahSariboz
"""

weights = [[0.1, 0.1, -0.3], #hurt?
           [0.1, 0.2, 0.0], # win?
           [0.0, 1.3, 0.1] ] # sad?



toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]


def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        print(b[i])
        output += (a[i] * b[i])
    return output


def vect_mat_mul(vect,matrix):
    assert(len(vect) == len(matrix))
    output = [0,0,0]
    for i in range(len(vect)):
        output[i] = w_sum(vect,matrix[i])
    return output


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred

pred = neural_network(input, weights)