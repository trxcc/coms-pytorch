import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

from torch.utils.data import DataLoader
from torch.autograd import Variable
from typing import Optional

from metrics import spearman_correlation

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

class ConservativeObjectiveTrainer:
    def __init__(self, model, config):
        self.config = config
        
        self.forward_lr = config["forward_lr"]
        # self.forward_lr_decay = config["forward_lr_decay"]
        self.n_epochs = config["n_epochs"]
        
        self.model = model
        
        ################## TODO: to be fixed ################
        try:
            self.forward_opt = torch.optim.Adam(model.parameters(),
                                    lr=config["forward_lr"])
        except:
            pass
        #####################################################
        self.train_criterion = lambda yhat, y: torch.sum(torch.mean((yhat-y)**2, dim=1)) 
        self.mse_criterion = nn.MSELoss()
        
        alpha = torch.tensor(config["alpha"])
        self.log_alpha = torch.nn.Parameter(torch.log(alpha))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config["alpha_lr"])

        self.overestimation_limit = config["overestimation_limit"]
        self.particle_lr = config["particle_lr"] * np.sqrt(np.prod(config["input_shape"]))
        self.particle_gradient_steps = config["particle_gradient_steps"]
        self.entropy_coefficient = config["entropy_coefficient"]
        self.noise_std = config["noise_std"]
        
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
        
    def obtain_x_neg(self, x, steps, **kwargs):
        
        # gradient ascent on the conservatism
        def gradient_step(xt):
            
            # shuffle the designs for calculating entropy
            indices = torch.randperm(xt.size(0))
            shuffled_xt = xt[indices]
            
            # entropy using the gaussian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            score = self.model(xt, **kwargs)

            # the conservatism of the current set of particles
            losses = self.entropy_coefficient * entropy + score
            
            # calculate gradients for each element separately
            grads = torch.autograd.grad(outputs=losses, inputs=xt, grad_outputs=torch.ones_like(losses))

            with torch.no_grad():
                xt.data = xt.data - self.particle_lr * grads[0].detach()
                xt.detach_()
                if xt.grad is not None:
                    xt.grad.zero_()
            return xt.detach()

        xt = torch.tensor(x, requires_grad=True).to(**tkwargs)
        
        for _ in range(steps):
            xt = gradient_step(xt)
            xt.requires_grad = True
        return xt
    
    def train_step(self, x, y, statistics):
        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x).to(**tkwargs)
        x, y = Variable(x, requires_grad=True), Variable(y, requires_grad=False)

        # calculate the prediction error and accuracy of the model
        d_pos = self.model(x)
        mse = F.mse_loss(d_pos.squeeze().to(**tkwargs), y.squeeze().to(**tkwargs))

        # calculate negative samples starting from the dataset
        x_neg = self.obtain_x_neg(x, self.particle_gradient_steps)
        # calculate the prediction error and accuracy of the model
        d_neg = self.model(x_neg)
        overestimation = d_pos[:, 0] - d_neg[:, 0]
        statistics[f"train/overestimation/mean"] = overestimation.mean()
        statistics[f"train/overestimation/std"] = overestimation.std()
        statistics[f"train/overestimation/max"] = overestimation.max()

        # build a lagrangian for dual descent
        alpha_loss = (self.alpha * self.overestimation_limit -
                    self.alpha * overestimation)
        statistics[f"train/alpha"] = self.alpha

        # loss that combines maximum likelihood with a constraint
        model_loss = mse + self.alpha * overestimation
        total_loss = model_loss.mean()
        alpha_loss = alpha_loss.mean()

        # calculate gradients using the model
        alpha_grads = torch.autograd.grad(alpha_loss, self.log_alpha, retain_graph=True)[0]
        model_grads = torch.autograd.grad(total_loss, self.model.parameters())

        # take gradient steps on the model
        with torch.no_grad():
            self.log_alpha.grad = alpha_grads
            self.alpha_opt.step()
            self.alpha_opt.zero_grad()

            for param, grad in zip(self.model.parameters(), model_grads):
                param.grad = grad
            self.forward_opt.step()
            self.forward_opt.zero_grad()
        
        return statistics
    
    def _evaluate_performance(self,
                              statistics,
                              epoch,
                              train_loader,
                              val_loader):
        self.model.eval()
        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            for batch_x, batch_y, in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y.reshape(-1, 1)), dim=0)
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)
            
            statistics[f"train/mse"] = train_mse.item() 
            # for i in range(self.n_obj):
            statistics[f"train/rank_corr"] = train_corr.item()
                
            print ('Epoch [{}/{}], MSE: {:}'
                .format(epoch+1, self.n_epochs, train_mse.item()))
        
        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y.reshape(-1, 1)), dim=0)
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))
            
            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)
            
            statistics[f"valid/mse"] = val_mse.item() 
            statistics[f"valid/rank_corr"] = val_corr.item()
                
            print ('Valid MSE: {:}'.format(val_mse.item()))
            
        return statistics
    
    def launch(self, 
               train_loader: Optional[DataLoader] = None,
               val_loader: Optional[DataLoader] = None,
               test_loader: Optional[DataLoader] = None):
        
        assert train_loader is not None 
        assert val_loader is not None 
        
        self.n_obj = 1
        iters = 0
        self.min_mse = float("inf")
        statistics = {}
        
        for epoch in range(self.n_epochs):
            
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                
                statistics = self.train_step(batch_x, batch_y, statistics)
        
            self._evaluate_performance(statistics, epoch,
                                        train_loader,
                                        val_loader)
            
        print(statistics)