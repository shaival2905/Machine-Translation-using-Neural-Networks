# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:39:54 2018

@author: shaival
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Bidirectional, Activation, RepeatVector
from keras.layers import Dot, Lambda, Multiply, Add
from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
import tensorflow as tf
import numpy as np

class Modelv1:
    
    def __init__(self, enc_seq_length, enc_unique_states, dec_seq_length, dec_unique_states, enc_layers=1, dec_layers=1, dense__prev_layers_neurons=[], lstm_units = 256, bidirectional=False, dropout=0, recurrent_dropout=0, bias_regularizer=None, kernel_regularizer=None, activity_regularizer=None, patience=3):
        self.enc_seq_length = enc_seq_length
        self.enc_unique_states = enc_unique_states
        self.dec_seq_length = dec_seq_length
        self.dec_unique_states = dec_unique_states
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dense__prev_layers_neurons = dense__prev_layers_neurons
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.early_stopping_monitor = EarlyStopping(patience=patience)
        self.tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, # To visualize model learning
                                       write_graph=True, write_images=True)
        self.dense__prev_layers_neurons.append(self.dec_unique_states)
        
    def getModel(self):
        
        self.encoder_inputs = Input(shape=(None, self.enc_unique_states), name='encoder_inputs')
        
        self.encoder = []
        self.encoder_outputs = []
        
        # Add encoder layers 
        for i in range(self.enc_layers-1):
            self.encoder.append(LSTM(self.lstm_units, 
                                     return_sequences=True, 
                                     recurrent_dropout=self.recurrent_dropout, 
                                     dropout = self.dropout, 
                                     bias_regularizer = self.bias_regularizer, 
                                     activity_regularizer = self.activity_regularizer, 
                                     kernel_regularizer=self.kernel_regularizer, 
                                     name="encoder"+str(i+1)))
            # Wrap Bidirectional layer if bidirectional is True
            if self.bidirectional:
                self.encoder[i] = Bidirectional(self.encoder[i])
            
        self.encoder.append(LSTM(self.lstm_units,
                                 return_sequences=True, 
                                 return_state=True, 
                                 recurrent_dropout=self.recurrent_dropout, 
                                 dropout = self.dropout, 
                                 bias_regularizer = self.bias_regularizer, 
                                 activity_regularizer = self.activity_regularizer, 
                                 kernel_regularizer=self.kernel_regularizer, 
                                 name="encoder"+str(self.enc_layers)))
        if self.bidirectional:
                self.encoder[self.enc_layers-1] = Bidirectional(self.encoder[self.enc_layers-1])
        
        # Get encoder outputs for each encoder layer
        for i in range(self.enc_layers):
            if i==0:
                self.encoder_outputs.append(self.encoder[i]((self.encoder_inputs)))
            else:
                self.encoder_outputs.append(self.encoder[i](self.encoder_outputs[i-1]))
        
        if self.bidirectional:
            _, self.state_h_l, self.state_h_r, self.state_c_l, self.state_c_r = self.encoder_outputs[self.enc_layers-1]           
            self.state_h = Concatenate()([self.state_h_l, self.state_h_r])
            self.state_c = Concatenate()([self.state_c_l, self.state_c_r])
            self.lstm_units = self.lstm_units*2
        else:
            _, self.state_h, self.state_c = self.encoder_outputs[self.enc_layers-1]
        
        # Calculate encoder states of last encoder cell which will be feeded to decoder first cell
        self.encoder_states = [self.state_h, self.state_c] 
        
        self.decoder_inputs = Input(shape=(None, self.dec_unique_states), name="decoder_inputs")
        self.decoder = []
        self.decoder_outputs = []
        
        # Add decoder layers
        for i in range(self.dec_layers-1):
            self.decoder.append(LSTM(self.lstm_units, 
                                     return_sequences=True, 
                                     recurrent_dropout=self.recurrent_dropout, 
                                     dropout = self.dropout, 
                                     bias_regularizer = self.bias_regularizer, 
                                     activity_regularizer = self.activity_regularizer, 
                                     kernel_regularizer=self.kernel_regularizer, 
                                     name="decoder"+str(i+1)))
        
        self.decoder.append(LSTM(self.lstm_units, 
                                 return_sequences=True, 
                                 return_state=True, 
                                 recurrent_dropout=self.recurrent_dropout, 
                                 dropout = self.dropout, 
                                 bias_regularizer = self.bias_regularizer, 
                                 activity_regularizer = self.activity_regularizer, 
                                 kernel_regularizer=self.kernel_regularizer, 
                                 name="decoder"+str(self.dec_layers)))
        
        # Get decoder outputs for each decoder layer
        for i in range(self.dec_layers):
            if i==0:
                self.decoder_outputs.append(self.decoder[i](self.decoder_inputs, initial_state=self.encoder_states))
            else:
                self.decoder_outputs.append(self.decoder[i](self.decoder_outputs[i-1]))
        
        
        self.decoder_dense = []
        self.dense_outputs = []
        self.dense_layers = len(self.dense__prev_layers_neurons)
        
        # Add fully connected layers
        for i in range(self.dense_layers):
            if i < self.dense_layers-1:
                self.decoder_dense.append(Dense(self.dense__prev_layers_neurons[i], 
                                                bias_regularizer = self.bias_regularizer, 
                                                activity_regularizer = self.activity_regularizer, 
                                                activation='relu', name="output_layer"+str(i+1)))
            else:
                self.decoder_dense.append(Dense(self.dense__prev_layers_neurons[i], 
                                                bias_regularizer = self.bias_regularizer, 
                                                activity_regularizer = self.activity_regularizer, 
                                                activation='softmax', name="output_layer"+str(i+1)))                
            
        # Get outputs of each fully connected layer
        for i in range(self.dense_layers):
            if i==0:
                self.dense_outputs.append(self.decoder_dense[i](self.decoder_outputs[self.dec_layers-1][0]))
            else:
                self.dense_outputs.append(self.decoder_dense[i](self.dense_outputs[i-1]))
        

        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.dense_outputs[self.dense_layers-1])
        
        return self.model
    
    def compileModel(self, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model
    
    def test(self):
        # Inference model used at the time of prediction
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        self.decoder_state_input_h = Input(shape=(self.lstm_units,))
        self.decoder_state_input_c = Input(shape=(self.lstm_units,))
        
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_outputs_inf = []
        for i in range(self.dec_layers):
            if i==0:
                self.decoder_outputs_inf.append(self.decoder[i](self.decoder_inputs, initial_state=self.decoder_states_inputs))
            else:
                self.decoder_outputs_inf.append(self.decoder[i](self.decoder_outputs_inf[i-1]))
        
        self.decoder_outputs_inf_final, self.state_h_inf, self.state_c_inf = self.decoder_outputs_inf[self.dec_layers-1]
        
        self.decoder_states_inf = [self.state_h_inf, self.state_c_inf]
        
        
        for i in range(self.dense_layers):
            if i==0:
                self.dense_outputs_inf = self.decoder_dense[i](self.decoder_outputs_inf_final)
            else:
                self.dense_outputs_inf = self.decoder_dense[i](self.dense_outputs_inf)
                
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.dense_outputs_inf] + self.decoder_states_inf)
        
    def train(self, train_x, train_y, batch_size, epochs, validation_split=0, test_x=None, test_y=None):
        if test_x is not None and test_y is not None:
            validation_data = [test_x, test_y] 
        else:
            validation_data = None
            
        self.model.fit(train_x, 
                       train_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split,
                       validation_data=validation_data)#,
                       #callbacks=[self.early_stopping_monitor, self.tensorboard])
        
        # Inference model used at the time of prediction
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        self.decoder_state_input_h = Input(shape=(self.lstm_units,))
        self.decoder_state_input_c = Input(shape=(self.lstm_units,))
        
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_outputs_inf = []
        for i in range(self.dec_layers):
            if i==0:
                self.decoder_outputs_inf.append(self.decoder[i](self.decoder_inputs, initial_state=self.decoder_states_inputs))
            else:
                self.decoder_outputs_inf.append(self.decoder[i](self.decoder_outputs_inf[i-1]))
        
        self.decoder_outputs_inf_final, self.state_h_inf, self.state_c_inf = self.decoder_outputs_inf[self.dec_layers-1]
        
        self.decoder_states_inf = [self.state_h_inf, self.state_c_inf]
        
        
        for i in range(self.dense_layers):
            if i==0:
                self.dense_outputs_inf = self.decoder_dense[i](self.decoder_outputs_inf_final)
            else:
                self.dense_outputs_inf = self.decoder_dense[i](self.dense_outputs_inf)
                
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.dense_outputs_inf] + self.decoder_states_inf)
        
        return self.model
    
    def decode_sequence(self, input_seq, target_token_index, reverse_target_char_index):
        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, self.dec_unique_states))

        target_seq[0, 0, target_token_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''
        l=0
        while not stop_condition:
            
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
    
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]

            decoded_sentence = decoded_sentence + " " + sampled_char
            l = l + 1

            if (sampled_char == '\n' or l >= self.dec_seq_length-2):
                stop_condition = True
    
            target_seq = np.zeros((1, 1, self.dec_unique_states))
            target_seq[0, 0, sampled_token_index] = 1.
    
            states_value = [h, c]
    
        return decoded_sentence
    
    def predict_data(self, encoder_input_data, target_seq, target_token_index, reverse_target_char_index, input_seq=None, verbose=False):
        c=0
        for seq_index in range(int(len(encoder_input_data)/5000)):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_enc_seq = encoder_input_data[seq_index: seq_index + 1]
            
            # print(input_seq)
            decoded = self.decode_sequence([input_enc_seq], target_token_index, reverse_target_char_index)
            inputseq = ""
            target=''
            
            if(target == decoded):
                c+=1
            
            if verbose:   
                for i in range(1,len(target_seq[seq_index])-1):
                    target = target + " " + target_seq[seq_index][i]
                print('-')  
                if input_seq is not None:
                    for state in input_seq[seq_index]:
                        inputseq = inputseq + " " + state
                    print('Input sequence:', inputseq)
                print('Expected: ' + target)
                print('  Output: ' + decoded)
                
        print(c,"correctly predicted from",len(encoder_input_data))
        
    def predict_single(self, x):       
        decoded = self.decode_sequence(x)
        print('  Output: ' + decoded)


class Modelv2:
    def __init__(self, enc_seq_length, enc_unique_states, dec_seq_length, dec_unique_states, lstm_units = 256, bidirectional=False, recurrent_dropout=0, kernel_regularizer=None, patience=15):
        self.enc_seq_length = enc_seq_length
        self.enc_unique_states = enc_unique_states
        self.dec_seq_length = dec_seq_length
        self.dec_unique_states = dec_unique_states
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.recurrent_dropout = recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.early_stopping_monitor = EarlyStopping(patience=patience)
        self.tensorboard = TensorBoard(log_dir='logs', histogram_freq=0,
                                       write_graph=True, write_images=True)
        self.repeator = RepeatVector(self.enc_seq_length)
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(10, activation = "tanh")
        self.densor2 = Dense(1, activation = "relu")
        self.activator = Activation('softmax', name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        self.dotor = Dot(axes = 1)
        self.output_layer = Dense(self.dec_unique_states, activation='softmax')
     
    
    def one_step_attention(self, a, s_prev, repeator):
        s_prev = self.repeator(s_prev) 
        concat = self.concatenator([a,s_prev]) 
        e = self.densor1(concat) 
        energies = self.densor2(e) 
        alphas = self.activator(energies) 
        context = self.dotor([alphas,a]) 
        return context

    def getModel(self):
        
        n_a = self.lstm_units
        n_s = n_a*2
        post_activation_LSTM_cell = LSTM(n_s, return_state = True)
        
        X = Input(shape=(self.enc_seq_length, self.enc_unique_states))
        
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0
        
        outputs = []
        
        a, state_h, state_h_rev, state_c, state_c_rev = Bidirectional(LSTM(n_a, return_sequences=True, return_state=True), input_shape=(None, self.enc_seq_length, n_a*2))(X)  

        for t in range(int(self.dec_seq_length)):
            
            
            context = self.one_step_attention(a, s, self.repeator)
            
            s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
            
            out = self.output_layer(s)
            
            outputs.append(out)
        
        def output(outputs):
            return tf.stack(outputs, axis=1)
        
        layer = Lambda(output)
        outputs = layer(outputs)
        self.model = Model([X,s0,c0],outputs)
        
        return self.model
    
    def compileModel(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model
    
    def train(self, train_x, train_y, batch_size, epochs, validation_split=None, test_x=None, test_y=None):
        
        s0 = np.zeros((len(train_x), self.lstm_units*2))
        c0 = np.zeros((len(train_x), self.lstm_units*2))
        
        s0_test = np.zeros((len(train_x), self.lstm_units*2))
        c0_test = np.zeros((len(train_x), self.lstm_units*2))
        
        self.model.fit([train_x, s0, c0], 
                       train_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data= [[test_x, s0_test, c0_test], test_y],
                       callbacks=[self.early_stopping_monitor, self.tensorboard])
        
        return self.model
