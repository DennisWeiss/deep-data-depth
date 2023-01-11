from enum import Enum
from typing import Callable
import torch
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from models import DeepDataDepthEncoder
import logging
import matplotlib.pyplot as plt
import numpy as np


class DeepDataDepthAnomalyDetector:
    Method = Enum('Method', [
        'VarianceMaximization',
        'KlDivFit'
    ])

    def __init__(self,
                 method: Method = Method.KLDivFit,
                 batch_size: int = 1000,
                 encoder_lr: float = 1e-3,
                 halfspace_optim_lr: float = 1e+3,
                 weight_decay: float = 0,
                 temp: float = 1e-1,
                 representation_dim: int = 32,
                 target_dist: Callable[[float], float] = lambda x: 8 * x,
                 iter: int = 40,
                 data_depth_iter: int = 20,
                 data_depth_computations: int = 20
                 ):
        """
        :param method (DeepDataDepthAnomalyDetector.Method, optional): Method to use for optimizing the data depth distribution (default: ``DeepDataDepthAnomalyDetector.Method.KLDivFit``)
        :param batch_size (int, optional): Batch size (default: ``1000``)
        :param encoder_lr (float, optional): Learning rate for encoder (default: ``1e-3``)
        :param halfspace_optim_lr (float, optional): Learning rate of the halfspace optimizer (default: ``1e+3``)
        :param weight_decay (float, optional): Weight decay (default: ``0``)
        :param temp (float, optional): Temperature value (default: ``1e-1``)
        :param representation_dim (int, optional): Representation dimension (default: ``32``)
        :param target_dist (Callable[[float], float], optional): Target distribution. Used only for KL divergence fit (default: ``f(x) = 8x``)
        :param iter (int, optional): Number of iterations of training the encoder (default: ``40``)
        :param data_depth_iter (int, optional): Number of iterations of the halfspace optimizer (default: ``20``)
        :param data_depth_computations (int, optional): Number of re-computations of the data depth value (default: ``20``)
        """
        self.X_train = None
        self.method = method
        self.batch_size = batch_size
        self.encoder_lr = encoder_lr
        self.halfspace_optim_lr = halfspace_optim_lr
        self.weight_decay = weight_decay
        self.temp = temp
        self.representation_dim = representation_dim
        self.target_dist = target_dist
        self.iter = iter
        self.data_depth_iter = data_depth_iter
        self.data_depth_computations = data_depth_computations

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def soft_tukey_depth(X_: torch.Tensor, X: torch.Tensor, Z: torch.Tensor, temp: float) -> torch.Tensor:
        X_new = X.repeat(X_.size(dim=0), 1, 1)
        X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
        X_diff = X_new - X_new_tr
        dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
        dot_products_normalized = dot_products.transpose(0, 1).divide(temp * Z.norm(dim=1))
        return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))

    @staticmethod
    def get_kl_divergence(soft_tukey_depths: torch.Tensor, f: Callable[[float], float], kernel_bandwidth: float = 0.05,
                          epsilon: float = 1e-3) -> torch.Tensor:
        delta = 0.005
        kl_divergence = torch.tensor(0)
        for x in torch.arange(0, 0.5, delta):
            val = torch.exp(torch.square(soft_tukey_depths - x).divide(
                torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))).mean()
            f_val = f(x)
            kl_divergence = kl_divergence.subtract(
                torch.multiply(torch.tensor(f_val * delta), torch.log(val.divide(f_val + epsilon))))
        return kl_divergence

    @staticmethod
    def draw_histogram(X: torch.Tensor, X_: torch.Tensor, z: torch.Tensor, temp:float, bins=100):
        soft_tukey_depths = DeepDataDepthEncoder.soft_tukey_depth(X, X_, z, temp)
        tukey_depth_histogram = plt.figure()
        plt.hist(soft_tukey_depths.detach().cpu().numpy(), bins=bins)
        tukey_depth_histogram.show()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        :param X_train: Training data
        :param y_train: Training labels
        :return: self
        """
        X_train = X_train[y_train == 0]

        self.X_train = X_train

        train_tensor = TensorDataset(torch.from_numpy(X_train).float())
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)
        train_loader_full = DataLoader(train_tensor, batch_size=X_train.shape[0], shuffle=False)

        n = X_train.shape[0]
        input_dim = X_train.shape[1]

        hidden_layer_dim = min(max(input_dim, self.representation_dim), 2 * self.representation_dim)

        self.encoder = DeepDataDepthEncoder(input_dim, hidden_layer_dim, self.representation_dim).to(self.device)
        self.encoder.train()

        optimizer_encoder = Adam(self.encoder.parameters(), lr=self.encoder_lr, weight_decay=self.weight_decay)

        best_z = 2 * torch.rand(n, self.representation_dim).to(self.device) - 1

        i = 0

        breaking = False

        while True:
            if (i + 1) % 5 == 0:
                logging.log(logging.INFO, f'Iteration {i + 1}/{self.iter}')
            for step_type in ['optimize_halfspace', 'optimize_encoder']:
                for step, (x,) in enumerate(train_loader):
                    if i >= self.iter:
                        breaking = True
                        break
                    x = x.to(self.device)
                    y = self.encoder(x)
                    y_detached = y.detach()

                    for step2, (x_full,) in enumerate(train_loader_full):
                        x_full = x_full.to(self.device)
                        y_full = self.encoder(x_full)
                        y_full_detached = y_full.detach()

                        if step_type == 'optimize_halfspace':
                            tukey_depths = self.soft_tukey_depth(
                                y_detached,
                                y_full_detached,
                                best_z[(step * self.batch_size):((step + 1) * self.batch_size)],
                                self.temp
                            )

                            for k in range(self.data_depth_computations):
                                z = 2 * torch.rand(n, self.representation_dim).to(self.device) - 1
                                optimizer_z = SGD([z], lr=self.halfspace_optim_lr)
                                for l in range(self.data_depth_iter):
                                    optimizer_z.zero_grad()
                                    current_tukey_depths = self.soft_tukey_depth(y_detached, y_full_detached, z,
                                                                                 self.temp)
                                    current_tukey_depths.sum().backward()
                                    optimizer_z.step()

                                current_tukey_depths = self.soft_tukey_depth(y_detached, y_full_detached, z, self.temp)

                                for l in range(tukey_depths.size(dim=0)):
                                    if current_tukey_depths[l] < tukey_depths[l]:
                                        tukey_depths[l] = current_tukey_depths[l].detach()
                                        best_z[(step * self.batch_size) + l] = z[l].detach()
                        elif step_type == 'optimize_encoder':
                            tukey_depths = self.soft_tukey_depth(y_full, y, best_z, self.temp)

                            optimizer_encoder.zero_grad()
                            if self.method == DeepDataDepthEncoder.Method.VarianceMaximization:
                                loss = torch.var(tukey_depths)
                            elif self.method == DeepDataDepthEncoder.Method.KLDivergence:
                                loss = self.get_kl_divergence(tukey_depths, self.target_dist)
                            loss.backward()
                            optimizer_encoder.step()
                            if (i + 1) % 5 == 0:
                                self.draw_histogram(
                                    y_detached,
                                    y_full_detached,
                                    best_z[(step * self.batch_size):((step + 1) * self.batch_size)].detach(),
                                    self.temp
                                )
                    i += 1 / 2
                if breaking:
                    break
            if breaking:
                break

        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Data
        :return: Anomaly scores
        """
        if self.X_train is None:
            raise Exception('Model not fitted yet')

        self.encoder.eval()

        train_data = TensorDataset(torch.from_numpy(self.X_train).float())
        train_loader = DataLoader(train_data, batch_size=self.X_train.shape[0], shuffle=False)

        test_data = TensorDataset(torch.from_numpy(x).float())
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

        scores = np.zeros(0)

        for step, (x_train,) in enumerate(train_loader):
            x_train = x_train.to(self.device)
            y_train = self.encoder(x_train)
            y_train_detached = y_train.detach()

            for step2, (x_test,) in enumerate(test_loader):
                x_test = x_test.to(self.device)
                y_test = self.encoder(x_test)
                y_test_detached = y_test.detach()

                best_z = 2 * torch.rand(x_test.shape[0], self.representation_dim).to(self.device) - 1
                tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, best_z, self.temp)

                for i in range(2 * self.data_depth_computations):
                    z = 2 * torch.rand(x_test.shape[0], self.representation_dim).to(self.device) - 1
                    optimizer_z = SGD([z], lr=self.halfspace_optim_lr)
                    for j in range(2 * self.data_depth_iter):
                        optimizer_z.zero_grad()
                        current_tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, z, self.temp)
                        current_tukey_depths.sum().backward()
                        optimizer_z.step()

                    current_tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, z, self.temp)

                    for j in range(current_tukey_depths.size(dim=0)):
                        if current_tukey_depths[j] < best_z[j]:
                            tukey_depths[j] = current_tukey_depths[j].detach()
                            best_z[j] = z[j].detach()

                scores = np.concatenate((scores, (0.5 - tukey_depths).cpu().numpy()))

        return scores
