import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
#import process_edited as pce
import process_GQ as pce

import tqdm
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt

def _make_mlp_layers(num_units):
    layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
        for in_features, out_features in zip(num_units, num_units[1:])
    ])
    return layers

class TabPFNDecoder(nn.Module):
    def __init__(self, embedding_size, n_bins, n_cats, n_nums, cards, hidden_size=128, num_layers=3):
        super().__init__()
        
        # The number of decoder layers should be defined
        decoder_layers = num_layers 
        
        # The decoder input will be the TabPFN embedding size
        self.decoders = _make_mlp_layers([embedding_size] + [hidden_size] * (decoder_layers - 1))
        
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        
        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins > 0 else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats > 0 else None
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums > 0 else None

    def forward(self, x):
        for layer in self.decoders:
            x = F.relu(layer(x))

        # Initialize a list to hold only the generated Tensor outputs
        outputs = []

        if self.n_nums > 0:
            num_output = self.nums_linear(x)
            outputs.append(num_output) # Only append if it's a Tensor

        if self.n_bins > 0:
            bins_output = self.bins_linear(x)
            outputs.append(bins_output) # Only append if it's a Tensor
            
        if self.n_cats > 0:
            # Note: cats_output is already a concatenated Tensor here
            cats_output = torch.cat([self.cats_linears[i](x) for i in range(self.n_cats)], dim=1)
            outputs.append(cats_output) # Only append if it's a Tensor

        # Concatenate ONLY the generated tensors
        return torch.cat(outputs, dim=1)


class DeapStack(nn.Module):
    ''' Simple MLP body. '''
    def __init__(self,  n_bins, n_cats, n_nums, cards, in_features, hidden_size, bottleneck_size, num_layers):
        super().__init__()       
        encoder_layers = num_layers >> 1
        decoder_layers = num_layers - encoder_layers - 1
        self.encoders = _make_mlp_layers([in_features] + [hidden_size] * encoder_layers)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.decoders = _make_mlp_layers([bottleneck_size] + [hidden_size] * decoder_layers)

        self.n_bins = n_bins
        self.n_cats = n_cats       
        self.n_nums = n_nums

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards])
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None




    def load_tabpfn_weights(self, tabpfn_encoder_state_dict, freeze=True):
        
        # 1. Load weights into the Encoder layers
        # TabPFN is a bit complex, but its feature extractor usually consists of an embedding MLP 
        # (similar to your 'encoders') and a Transformer. We will only initialize the MLP part.
        
        encoder_state = self.encoders.state_dict()

        # Try to map the first few layers directly
        for name, param in self.encoders.named_parameters():
            # Example mapping: 
            # DeapStack.encoders.0.weight -> tabpfn_encoder_state_dict['feature_extractor.0.weight']
            tabpfn_key = f"feature_extractor.{name}" # This key will vary!
            
            if tabpfn_key in tabpfn_encoder_state_dict:
                param.data.copy_(tabpfn_encoder_state_dict[tabpfn_key].data)
                
        print("DeapStack Encoder weights initialized with TabPFN Feature Extractor.")
        
        # 2. Freeze the initialized layers (RECOMMENDED for stability and speed)
        if freeze:
            for param in self.encoders.parameters():
                param.requires_grad = False
            for param in self.bottleneck.parameters():
                param.requires_grad = False
            print("DeapStack Encoder and Bottleneck layers are now frozen.")    

    def forward_pass(self, x):
        for encoder_layer in self.encoders:
            x = F.relu(encoder_layer(x))
        x = b = self.bottleneck(x)
        for decoder_layer in self.decoders:
            x = F.relu(decoder_layer(x))
        return [b, x]

    def forward(self, x):
        outputs = dict()
        
        num_min_values, _ = torch.min(x[:,self.n_bins+self.n_cats:self.n_bins+self.n_cats+self.n_nums], dim=0)
        num_max_values, _ = torch.max(x[:,self.n_bins+self.n_cats:self.n_bins+self.n_cats+self.n_nums], dim=0)
        
        decoder_output = self.forward_pass(x)[1]
        
        if self.bins_linear:
            outputs['bins'] = self.bins_linear(decoder_output)

        if self.cats_linears:
            outputs['cats'] = [linear(decoder_output) for linear in self.cats_linears]            
            
        if self.nums_linear:
            before_threshold = self.nums_linear(decoder_output)
            outputs['nums'] = before_threshold
            
            for col in range(len(num_min_values)):
                outputs['nums'][:,col] = torch.where(before_threshold[:,col] < num_min_values[col], num_min_values[col], before_threshold[:,col])     
                outputs['nums'][:,col] = torch.where(before_threshold[:,col] > num_max_values[col], num_max_values[col], before_threshold[:,col]) 
                                                     
        return outputs

    def featurize(self, x):
        return self.forward_pass(x)[0]
    
    def decoder(self, latent_feature, num_min_values, num_max_values):
        decoded_outputs = dict()

        for layer in self.decoders:
            x = F.relu(layer(latent_feature))
        last_hidden_layer = x
        
        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(last_hidden_layer)

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(last_hidden_layer) for linear in self.cats_linears]            
            
        if self.nums_linear:
            d_before_threshold = self.nums_linear(last_hidden_layer)
            decoded_outputs['nums'] = d_before_threshold
            
            for col in range(len(num_min_values)):
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] < num_min_values[col], num_min_values[col], d_before_threshold[:,col])     
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] > num_max_values[col], num_max_values[col], d_before_threshold[:,col]) 
                
        return decoded_outputs

    



def auto_loss(data, output, n_bins, n_nums, n_cats, cards):
    """
    Calculates the combined loss for Binary, Categorical, and Numerical features.
    
    data: Target tensor (Binary labels, Categorical labels, Numerical values)
    output: Reconstructed tensor (Binary logits, Categorical logits (concatenated), Numerical values)
    """
    total_loss = 0.0
    current_input_idx = 0  # Tracks position in the TARGET tensor (data)
    current_output_idx = 0 # Tracks position in the RECONSTRUCTED tensor (output)

    # 1. Binary Loss (BCEWithLogits)
    if n_bins > 0:
        bin_output = output[:, current_output_idx : current_output_idx + n_bins]
        bin_target = data[:, current_input_idx : current_input_idx + n_bins]
        
        # Use BCEWithLogitsLoss for stability
        total_loss += F.binary_cross_entropy_with_logits(bin_output, bin_target, reduction='mean')
        
        current_input_idx += n_bins
        current_output_idx += n_bins

    # 2. Categorical Loss (CrossEntropyLoss - CE)
    if n_cats > 0:
        for card in cards:
            # Output: Sliced logits (size = card)
            cat_output = output[:, current_output_idx : current_output_idx + card]
            
            # Target: Single column of integer labels (size = 1) - Must be LongTensor
            cat_target_label = data[:, current_input_idx].long() 
            
            # CrossEntropyLoss expects logits and integer targets
            total_loss += F.cross_entropy(cat_output, cat_target_label, reduction='mean')
            
            # Target moves by 1 column (the label), Output moves by 'card' columns (the logits)
            current_input_idx += 1
            current_output_idx += card

    # 3. Numerical Loss (MSE Loss)
    if n_nums > 0:
        num_output = output[:, current_output_idx : current_output_idx + n_nums]
        num_target = data[:, current_input_idx : current_input_idx + n_nums]
        
        total_loss += F.mse_loss(num_output, num_target, reduction='mean')
        
    return total_loss


def sigmoid_threshold(logits):
    sigmoid_output = torch.sigmoid(logits)
    threshold_output = torch.where(sigmoid_output > 0.5, torch.tensor(1), torch.tensor(0))
    return threshold_output

def softmax_with_max(predictions):
    # Applying softmax function
    probabilities = F.softmax(predictions, dim=1)
    
    # Getting the index of the maximum element
    max_indices = torch.argmax(probabilities, dim=1)
    
    return max_indices

def train_autoencoder(df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold):
    parser = pce.DataFrameParser().fit(df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32'))

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']

    DS = DeapStack(n_bins, n_cats, n_nums, cards, data.shape[1], hidden_size=128, bottleneck_size=data.shape[1], num_layers=3)

    optimizer = Adam(DS.parameters(), lr=lr, weight_decay=weight_decay)

    tqdm_epoch = tqdm.notebook.trange(n_epochs)

    losses = []
    batch_size
    all_indices = list(range(data.shape[0]))

    for epoch in tqdm_epoch:
      batch_indices = random.sample(all_indices, batch_size)

      output = DS(data[batch_indices,:])

      l2_loss = auto_loss(data[batch_indices,:], output, n_bins, n_nums, n_cats, cards)
      optimizer.zero_grad()
      l2_loss.backward()
      optimizer.step()

      gc.collect()
      torch.cuda.empty_cache()

      # Print the training loss over the epoch.
      losses.append(l2_loss.item())

      tqdm_epoch.set_description('Average Loss: {:5f}'.format(l2_loss.item()))
    
    num_min_values, _ = torch.min(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)
    num_max_values, _ = torch.max(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)

    latent_features = DS.featurize(data)
    output = DS.decoder(latent_features, num_min_values, num_max_values)

    return (DS.decoder, latent_features, num_min_values, num_max_values)
