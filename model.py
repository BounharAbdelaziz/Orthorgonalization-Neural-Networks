
import torch
import torch.nn as nn

class Network(nn.Module):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def __init__( self, n_inputs=6, n_outputs=6, n_units=64, n_layers=2):

    super(Network, self).__init__() 
    self.net = []
    for i in range(n_layers):
      self.net.append(nn.Linear(n_inputs, n_units))
      n_inputs = n_units
    # output layer
    self.net.append(nn.Linear(n_inputs, n_outputs))
    self.net = nn.Sequential(*self.net)

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    
    out = self.net(x)
    return out

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

class CosineLoss(nn.Module):
    def __init__(self, use_torch_cosine_embedding_loss=False):
        super(CosineLoss, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss()
        self.eps = 1e-8
        self.use_torch_cosine_embedding_loss = use_torch_cosine_embedding_loss

    def cosine_similarity(self, x,y):
      dot_product = torch.dot(x,y)
      product_norm = torch.norm(x) * torch.norm(y)
      return dot_product/product_norm

    def forward(self, x, y):        
        loss = 0
        for x_i,y_i in zip(x, y):
          if self.use_torch_cosine_embedding_loss:
            loss = loss + torch.abs(1-self.criterion(x_i, y_i, torch.ones_like(x_i[0]))) # cosine embedding loss = 1-cos(x,y) so we need to subtract it from 1
          else:
            loss = loss + self.cosine_similarity(x_i,y_i)
          print(f'[INFO] loss = {loss}')
        return loss + self.eps