#PyTorch v1.13.0+cu117
#CUDA device available: True
#1 devices available
#device = cuda
#isnotebook = True
#isgooglecolab = False
#shell = ZMQInteractiveShell

layer_type = ["transfenc", "dense", "dense"]
Seed = 42
num_units = [1, 1, 256, 512]
activation = ["relu", "relu", "relu"]
dropout = [0.1, 0.1, 0.1]
usebias = [True, True, True, True]
batch_size = 4048
T_Length = 240
K_Length = 8
D_Length = 1
H1 = 240
W1 = 1
conv_input_size = (240, 1)
input_size = 240
output_size = 1
hn1 = 1

transf_nhead = [1]
transf_ff_dim = [256]
transf_l_norm = [1e-05]
transf_drp = [0.1]
transf_actv = ['relu']
l2_lamda = 0.05
mu = 0.99

PrintInfoEverynEpochs = 1

train_best_loss = 0.8763056397438049
valid_best_loss = 0.7889122366905212
ReluAlpha = 0
EluAlpha = 0
valid_metric1 = 0.7086989985392091
valid_metric2 = 0.9486774270136302
valid_metric3 = 0.7084859187371364
valid_best_metric1 = 0.7086989985392091
valid_best_metric2 = 0.9486774270136302
valid_best_metric3 = 0.7084859187371364
