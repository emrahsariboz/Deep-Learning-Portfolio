# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:37:51 2020

@author: EmrahSariboz
"""
import numpy as np

class neural_network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        self.acti_func = lambda x: ( 1 / ( 1 + np.exp(-x)))
        
        self.weights_i_h = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        self.weights_h_o = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                            (self.hidden_nodes, self.output_nodes))
    

    def train(self, features, target):
        