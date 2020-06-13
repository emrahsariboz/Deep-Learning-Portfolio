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
        
        for X, y in zip(features, target):
            final_outputs, hidden_outputs = self.forward_pass(X)
        
        
    
    def forward_pass(self, X):
        
        hidden_input= np.dot(X, self.weights_i_h)
        hidden_output = self.acti_func(hidden_input)
        
        
        #Output layer
        
        final_inputs = np.dot(hidden_output, self.weights_h_o)
        final_outputs = self.acti_func(final_inputs)
        
        return final_outputs, hidden_output
    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, 
                        delta_weights_h_o):
        
        error = 