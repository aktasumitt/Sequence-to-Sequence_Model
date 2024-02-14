# Basic_CNN_Model:

## Introduction:

 In this project, I aimed to train a seq2seq model using basic Encoder-Decoder architecture with Bahdanau Attention and with custom sequence Dataset.

## Dataset:
 - I used the custom dataset for this project.
 - The input data consists of arrays of random integers ranging from 1 to 15, with a maximum length of 10.
 - The output is the reverse of the input array. For example, if the input is [1,2,3,4,5], the output would be [5,4,3,2,1].
 - The aim of model training is to be able to find the reverse of a given list at the end.
 - Zero padding is applied to the end of the lists with a length of 10.
 - I randomly split the dataset into training, validation, and test sets with ratios of (0.5, 0.25, 0.25) respectively.

## Model:

  - In this model, I used a very basic encoder-decoder architecture, both consisting of a single LSTM layer, with a Bahdanau attention model in between.
  -  After the input encoder values are passed through an embedding layer, they are fed into the encoder, and then the encoder LSTM states become the initial states for the decoder LSTM.
  -   The initial_decoder_input and initial_context (both are zero tensors) are concatenated and passed through the embedding layer.
  -   This embedding output becomes the input for the decoder. 
  -   Subsequently, the decoder hidden state and the encoder output are sent to the Bahdanau attention to obtain a context vector.
  -   The context vector, along with the previous decoder output, is concatenated to become the new input for the next decoder step.
  -   This way, the decoder operates for a maximum length number of steps and generates the next output at each step.
  -   Look for more details: https://arxiv.org/abs/1409.0473 and https://arxiv.org/abs/1409.3215

## Train:
  - I chose Adam optimizer with a learning rate of 0.001 and used CrossEntropyLoss as the loss function. I trained the model for 20  
  epochs.

## Results:
- After 20 epochs, the model achieved approximately 97% accuracy on both the training, validation, and test sets.

## Usage: 
- You can train the model by setting Just "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Then you can predict the sequneces placed in the Prediction folder by setting the "Load_Checkpoint" and "Prediction" values to "True" in the config file.




