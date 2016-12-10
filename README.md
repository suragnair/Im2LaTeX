# Im2LaTeX
An implementation of the Show, Attend and Tell paper in TensorFlow, for the OpenAI Im2LaTeX suggested problem.

- The crux of the model is contained in cnn_enc_gru_dec_attn.py that uses the embedding attention decoder from TensorFlow to attend on the output of the CNN.

- test_cnn_enc_gru_dec_attn.py has a method to extract the attention coefficients for each time step, for which I have overwritten some TensorFlow functions from seq2seq.py.

- An example of attention map over an input LaTeX formula image, as the decoder outputs each token.

![alt tag](https://raw.githubusercontent.com/suragnair/Im2LaTeX/master/attn)
