import torch
import numpy as np

from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm

from ..core import Logger
from ..entities import SNGPInfo
from typing import Tuple


class SNGP(nn.Module):
    def __init__(
            self,
            out_features: int,
            backbone: nn.Module,
            backbone_output_features: int = 768,
            num_inducing: int = 1024,
            momentum: float = 0,
            ridge_penalty: float = 1e-6
    ):
        
        super().__init__()
        self.out_features = out_features
        self.num_inducing = num_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty

        # Random Fourier features (RFF) layer
        random_fourier_feature_layer = nn.Linear(backbone_output_features, num_inducing)
        random_fourier_feature_layer.weight.requires_grad_(False)
        random_fourier_feature_layer.bias.requires_grad_(False)
        nn.init.normal_(random_fourier_feature_layer.weight, mean=0.0, std=1.0)
        nn.init.uniform_(random_fourier_feature_layer.bias, a=0.0, b=2 * np.pi)

        # Apply spectral normalization
        random_fourier_feature_layer = spectral_norm(random_fourier_feature_layer)        

        self.rff = nn.Sequential(backbone, random_fourier_feature_layer)

        # RFF approximation reduces the GP to a standard Bayesian linear model,
        # with beta being the parameters we wish to estimate by maximising
        # p(beta | D). To this end p(beta) (the prior) is gaussian so the loss
        # can be written as a standard MAP objective
        self.beta = nn.Linear(num_inducing, out_features, bias=False)
        nn.init.normal_(self.beta.weight, mean=0.0, std=1.0)

        # RFF precision and covariance matrices
        self.register_buffer('is_fit', torch.tensor(False))
        self.is_fit = torch.tensor(False)

        self.register_buffer('max_variance', torch.ones(1))

        self.covariance = Parameter(
            self.ridge_penalty * torch.eye(num_inducing),
            requires_grad=False,
        )

        self.precision_initial = self.ridge_penalty * torch.eye(
            num_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )
        
        if torch.cuda.is_available():
            self = self.cuda()
            

            
    def forward(self, X, with_variance=False, update_precision=False):
        features = torch.cos(self.rff(X))

        if update_precision:
            self.update_precision_(features)

        logits = self.beta(features)

        if not with_variance:
            return logits, None
        else:
            if not self.is_fit:
                raise ValueError(
                    "`update_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(features[:, None, :], (features @ self.covariance)[:, :, None], ).reshape(-1)

            return logits, variances           
            
            
            

    def train_model(self,
              dataset: torch.utils.data.Dataset,
              epochs: int = 10,
              batch_size: int = 256,
              lr: float = 2e-5,
              weight_decay: float = 0
              ) -> None:

        # Set into training mode
        self.train()

        # Optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Losses
        celoss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')
        l2 = lambda betas: celoss(betas, betas.new_zeros(betas.shape))

        # Training batch
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []

        for i in range(epochs):
            
            for j, (x_batch, y_batch) in enumerate(data_loader):
                
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    x_batch = tuple(x.cuda() for x in x_batch)
                    y_batch = y_batch.cuda()
                
                if i == (epochs - 1):

                    pred, var = self(x_batch, with_variance=False, update_precision=True)
                    
                else:
                    
                    pred, _ = self(x_batch, with_variance=False, update_precision=False)
                
                
                neg_log_likelihood: Tensor = celoss(pred, y_batch)
                
                betas = self.beta.weight
                l2_loss = 0.5 * l2(betas)
                
                # − log p(β|D) = − log p(D|β) + 0.5*||β||2
                neg_log_posterior: Tensor = neg_log_likelihood + l2_loss
                losses.append(neg_log_posterior.item())
                
                optimizer.zero_grad()
                
                neg_log_posterior.backward()

                optimizer.step()
                

            print('Training in progress... ', int(((i+1)/epochs)*100), '%  Mean Loss... ', np.mean(losses))
            
        self.update_covariance()
        

    def predict(self, dataset: torch.utils.data.Dataset, batch_size: int = 256) -> SNGPInfo:

        # Testing batch
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        
        probs_list = []
        variance_list = []

        with torch.no_grad():
            
            for j, (x_batch, y_batch) in enumerate(data_loader):

                if torch.cuda.is_available():
                    x_batch = tuple(x.cuda() for x in x_batch)
                    y_batch = y_batch.cuda()
                                
                preds, variances = self(x_batch, with_variance=True, update_precision=False)

                logits = self.mean_field_logits(preds, variances)
   
                
                probs = logits.detach().cpu().softmax(dim=1).numpy()
                probs_list.append(probs)
                variance_list.append(variances.detach().cpu().numpy())
                
                print('Inference in progress... ', int(((j+1)/len(data_loader))*100), '%')
                       
        mean = np.concatenate(probs_list)  # type: ignore      
        decision = np.argmax(mean, axis = 1)
        variance = np.concatenate(variance_list)  # type: ignore
        deviation = np.sqrt(variance)
            

        return SNGPInfo(decision=decision,
                       mean=mean,
                       deviation=deviation,
                       variance=variance,
                       )
        
    
    def reset_precision(self) -> None:
        
        self.precision = self.precision_initial.detach()

    def update_precision_(self, features) -> None:
        
        # This assumes that all classes share a precision matrix like in
        # https://www.tensorflow.org/tutorials/understanding/sngp

        # The original SNGP paper defines precision and covariance matrices on a
        # per class basis, however this can get expensive to compute with large
        # output spaces
        
        with torch.no_grad():
            if self.momentum < 0:
                # self.precision = identity => self.precision = identity + features.T @ features
                self.precision = Parameter(self.precision + features.T @ features)
            else:
                self.precision = Parameter(self.momentum * self.precision +
                                           (1 - self.momentum) * features.T @ features)
                

    def update_precision(self, X: Tensor) -> None:
        with torch.no_grad():
            features = torch.cos(self.rff(X))
            self.update_precision_(features)
            

    def update_covariance(self) -> None:
        if not self.is_fit:
            # The precision matrix is positive definite and so we can use its cholesky decomposition to more
            # efficiently compute its inverse (when num_inducing is large)
            try:
                L = torch.linalg.cholesky(self.precision)
                self.covariance = Parameter(self.ridge_penalty * L.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)
                print('Cholesky')
            except:
                self.covariance = Parameter(self.ridge_penalty * self.precision.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)
                print('Standard inversion')
        else:
            print(f'No inversion, already fit')

    def reset_covariance(self) -> None:
        self.is_fit = torch.tensor(False)
        self.covariance.zero_()
        
    @staticmethod    
    def mean_field_logits(logits: Tensor, variances: Tensor, mean_field_factor: float = np.pi/8.) -> Tensor:
        
        logits_scale = (1.0 + variances * mean_field_factor) ** 0.5
        
        if len(logits.shape) > 1:
            logits_scale = logits_scale[:, None]

        return logits/logits_scale
