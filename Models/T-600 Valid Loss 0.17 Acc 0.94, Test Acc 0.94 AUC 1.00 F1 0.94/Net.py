#Custom Transformer Encoder
class Custom_Tranformer_Encoder(nn.Module):
    """A custom PyTorch implementation of a Transformer Encoding layer."""
    def __init__(self, seq_size, input_size, num_heads, ff_dim, epsilon = 1e-6, dropout = 0, activation = "relu"):
        super(Custom_Tranformer_Encoder, self).__init__()
        self.seq_size = seq_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.epsilon = epsilon
        self.dropout = dropout
        self.activation = activation
        
        self.layers = nn.ModuleList([
            nn.LayerNorm(self.input_size, eps = self.epsilon, elementwise_affine = True),
            torch.nn.MultiheadAttention(self.input_size, self.num_heads, dropout = self.dropout, bias = True, add_bias_kv = False, add_zero_attn = False, kdim = None, vdim = None, batch_first = True),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.input_size, eps = self.epsilon, elementwise_affine = True),
            nn.Conv1d(in_channels = self.seq_size, out_channels = self.ff_dim, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = True),
            self.GetActivationLayer(),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels = self.ff_dim, out_channels = self.seq_size, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = True)
        ])
        
    def forward(self, inputs):
        """Take a PyTorch Tensor input and use the forward direction of the Transformer to get an output of the same shape."""
        x_out = self.layers[0](inputs)
        x_out = self.layers[1](x_out, x_out, x_out)[0]
        x_out = self.layers[2](x_out)        
        res = torch.add(x_out, inputs)
        x_out = self.layers[3](res)
        x_out = self.layers[4](x_out)
        x_out = self.layers[5](x_out)
        x_out = self.layers[6](x_out)
        x_out = self.layers[7](x_out)
        result = torch.add(x_out, res)
        return(result)
    
    def GetActivationLayer(self):
        Result = None
        if (self.activation == "relu"): #Not differentiable at 0. Doesn't need Greedy layer-wise pretraining (Hinton) because it doesn't suffer from vanishing gradient
            Result = nn.ReLU()
        elif (self.activation == "relu6"):
            Result = nn.ReLU6()
        elif (self.activation == "elu"): #Like ReLu but allows values to be negative, so they can be centred around 0, also potential vanishing gradient on the left side but doesn't matter
            Result = nn.ELU(alpha = 0.1) #alpha: Slope on the left side
        elif (self.activation == "tanh"): #Suffers from Vanishing Gradient
            Result = nn.Tanh()
        elif (self.activation == "sigmoid"): #Suffers from Vanishing Gradient
            Result = nn.Sigmoid() #Result isn't centred around 0. Maximum derivative: 0.25
        return Result
        
class Net(nn.Module):
    def __init__(self, T, K, num_units, activation, usebias, dropout, EluAlpha, ReluAlpha, transf_nhead, transf_ff_dim, transf_l_norm, transf_drp = 0.1, transf_actv = "relu"):
        super(Net, self).__init__()
        self.T = T
        self.K = K
        self.num_units = num_units
        self.activation = activation
        self.usebias = usebias
        self.dropout = dropout
        self.EluAlpha = EluAlpha
        self.ReluAlpha = ReluAlpha
        self.transf_nhead = transf_nhead
        self.transf_ff_dim = transf_ff_dim
        self.transf_l_norm = transf_l_norm
        self.transf_drp = transf_drp
        self.transf_actv = transf_actv
                
        self.layers = nn.ModuleList([
            Custom_Tranformer_Encoder(
                seq_size = self.T,
                input_size = self.num_units[0],
                num_heads = self.transf_nhead[0],
                ff_dim = self.transf_ff_dim[0],
                epsilon = self.transf_l_norm[0],
                dropout = self.transf_drp[0],
                activation = self.transf_actv[0]
            ),
            self.GetActivationLayer(0),
            nn.Dropout(p = self.dropout[0], inplace = False),
            nn.Linear(in_features = (self.T * self.num_units[1]), out_features = self.num_units[2], bias = self.usebias[1]),
            self.GetActivationLayer(1),
            nn.Dropout(p = self.dropout[1], inplace = False),
            nn.Linear(in_features = self.num_units[2], out_features = self.num_units[3], bias = self.usebias[2]),
            self.GetActivationLayer(2),
            nn.Dropout(p = self.dropout[2], inplace = False),
            nn.Linear(in_features = self.num_units[3], out_features = self.K, bias = self.usebias[3])
        ])
    
    def forward(self, x):
        out = self.layers[0](x)
        out = out.view(out.shape[0], -1)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.layers[3](out)
        out = self.layers[4](out)
        out = self.layers[5](out)
        out = self.layers[6](out)
        out = self.layers[7](out)
        out = self.layers[8](out)
        out = self.layers[9](out)        
        return out
    
    def GetActivationLayer(self, layer):
        Result = None
        if (self.activation[layer] == "relu"): #Not differentiable at 0. Doesn't need Greedy layer-wise pretraining (Hinton) because it doesn't suffer from vanishing gradient
            Result = nn.LeakyReLU(self.ReluAlpha) if self.ReluAlpha != 0 else nn.ReLU() #alpha: Controls the angle of the negative slope
        elif (self.activation[layer] == "relu6"):
            Result = nn.ReLU6()
        elif (self.activation[layer] == "elu"): #Like ReLu but allows values to be negative, so they can be centred around 0, also potential vanishing gradient on the left side but doesn't matter
            Result = nn.ELU(alpha = self.EluAlpha) #alpha: Slope on the left side
        elif (self.activation[layer] == "tanh"): #Suffers from Vanishing Gradient
            Result = nn.Tanh()
        elif (self.activation[layer] == "sigmoid"): #Suffers from Vanishing Gradient
            Result = nn.Sigmoid() #Result isn't centred around 0. Maximum derivative: 0.25
        return Result
print("Done")