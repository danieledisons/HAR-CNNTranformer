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
batch_size = 512
T_Length = 720
K_Length = 8
D_Length = 1
H1 = 720
W1 = 1
conv_input_size = (720, 1)
input_size = 720
output_size = 1
hn1 = 1

transf_nhead = [1]
transf_ff_dim = [2048]
transf_l_norm = [1e-05]
transf_drp = [0.1]
transf_actv = ['relu']
l2_lamda = 0.05
mu = 0.99

PrintInfoEverynEpochs = 1

train_best_loss = 0.21839287877082825
valid_best_loss = 0.14801497757434845
ReluAlpha = 0
EluAlpha = 0
valid_metric1 = 0.9495805868415438
valid_metric2 = 0.9980280714143612
valid_metric3 = 0.9495628374563839
valid_best_metric1 = 0.9495805868415438
valid_best_metric2 = 0.9980280714143612
valid_best_metric3 = 0.9495628374563839
