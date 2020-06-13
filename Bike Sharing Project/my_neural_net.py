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
        
        self.sigmoid_output_derivative = lambda x: (x * (1 - x))

    def train(self, features, target):
        
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, target):
            final_outputs, hidden_outputs = self.forward_pass(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        
    def MSE(y, Y):
        return np.mean((y-Y)**2)
    
    
    def forward_pass(self, X):
        
        hidden_input= np.dot(X, self.weights_i_h)
        hidden_output = self.acti_func(hidden_input)
        
        
        #Output layer
        
        final_inputs = np.dot(hidden_output, self.weights_h_o)
        final_outputs = self.acti_func(final_inputs)
        
        return final_outputs, hidden_output
    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, 
                        delta_weights_h_o):
        
        error = self.MSE(y - final_outputs)
        
        output_error_term = np.dot(error, self.weights_h_o)
        output_error_delta = output_error_term * self.sigmoid_output_derivative(error)
        
        hidden_error_term = np.dot(output_error_delta, self.weights_i_h)
        hidden_error_delta = hidden_error_term * self.sigmoid_output_derivative(output_error_term)
        
        delta_weights_h_o += np.dot(final_outputs, output_error_delta)
        delta_weights_i_h += np.dot(hidden_outputs, hidden_error_delta)
        
        return delta_weights_i_h, delta_weights_h_o
    
    def update_weights(self,delta_weights_i_h, delta_weights_h_o, n_records ):
        
        self.weights_hidden_to_output += delta_weights_h_o * self.learning_rate 
        self.weights_input_to_hidden += delta_weights_i_h * self.learning_rate 



















