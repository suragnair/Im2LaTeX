# Im2LaTeX
An implementation of the Show, Attend and Tell paper in TensorFlow, for the OpenAI Im2LaTeX suggested problem.

The crux of the model is contained in cnn_enc_gru_dec_attn.py that uses the embedding attention decoder from TensorFlow to attend on the output of the CNN.
