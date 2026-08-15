"""Models used for reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import Normalizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import warnings
from .data import LASSubset, Mutator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from tqdm import tqdm


class MultiLayerPerceptron(nn.Module):
    """Feed-forward neural network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_nodes: list[int] = [32, 64, 128, 64],
        dropout: float = 0.0,
    ) -> None:
        """Initialise the network.

        Args:
            input_size: Number of input features.
            output_size: Number of regression targets.
            layer_nodes: Hidden layer widths.
            dropout: Dropout probability applied after each hidden layer.
        """
        super(MultiLayerPerceptron, self).__init__()
        
        # Fully connected layers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_size, layer_nodes[0]))
        for i in range(1, len(layer_nodes)):
            self.fcs.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))
        self.fcs.append(nn.Linear(layer_nodes[-1], output_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            The network output tensor.
        """
        for fc in self.fcs[:-1]:
            x = torch.relu(fc(x))
            x = self.dropout(x)
        x = self.fcs[-1](x)
        return x


    def fit(
        self,
        train_dataset: LASSubset,
        val_dataset: LASSubset,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        use_gpu: bool = False,
        verbose: bool | int = False,
        mutator: Mutator | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_dataset: Training subset.
            val_dataset: Validation subset.
            criterion: Loss function used for optimisation.
            optimizer: Optimiser used for parameter updates.
            num_epochs: Number of training epochs.
            use_gpu: Whether to move the model to a GPU backend when available.
            verbose: Verbosity flag; ``2`` enables detailed progress messages.
            mutator: Optional data mutator applied to training inputs each epoch.

        Returns:
            A history dictionary containing per-epoch training and validation loss.
        """
        # Initialise history dictionary
        hist = {'train_loss': [], 'val_loss': []}
        updates = 0

        # move the model and data to the appropriate device
        if use_gpu:
            if torch.cuda.is_available():
                print("Using CUDA backend for PyTorch")
                self.to('cuda')
            elif torch.backends.mps.is_available():
                print("Using Apple MPS backend for PyTorch")
                self.to('mps')
            else:
                print("Using CPU backend for PyTorch")
                self.to('cpu')

        # Set model to training mode
        self.train()
        for epoch in tqdm(range(num_epochs), desc=f"Training ") if verbose else range(num_epochs):
            running_train_loss = 0.0
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5000, shuffle=False)
            for i, (inputs, labels) in enumerate(train_loader):    
                # Move tensors to the appropriate device (GPU or CPU)
                if mutator is not None:
                    inputs_np = inputs.numpy()
                    inputs_np = Normalizer("l1").fit_transform(mutator.mutate(inputs_np, epoch, num_epochs))
                    inputs = torch.from_numpy(inputs_np)
                else:
                    input_np = inputs.numpy()
                    input_np = Normalizer("l1").fit_transform(input_np)
                    inputs = torch.from_numpy(input_np)
                inputs, labels = inputs.to(torch.float32).to(next(self.parameters()).device), labels.to(torch.float32).to(next(self.parameters()).device)
                if epoch == 0 and i == 0 and verbose==2:
                    print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
                    print(f"Model device: {next(self.parameters()).device}")

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                inputs = inputs.float()
                labels = labels.float()

                outputs = self(inputs)
                loss: torch.Tensor = criterion(outputs, labels)  

                # Backward pass and optimize step
                loss.backward()
                optimizer.step()

                # Accumulate training loss
                running_train_loss += loss.item()
                updates += 1

            # Calculate average training loss for the epoch
            avg_train_loss = running_train_loss / len(train_loader)
            hist['train_loss'].append(avg_train_loss)

            # Calculate validation loss
            running_val_loss = 0.0
            self.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(torch.float32).to(next(self.parameters()).device), labels.to(torch.float32).to(next(self.parameters()).device)

                    inputs = inputs.float()
                    labels = labels.float()

                    # Forward pass
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            hist['val_loss'].append(avg_val_loss)
            
            # Print statistics
            if verbose==2:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}')

            # Switch back to training mode
            self.train()
            # tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}") if verbose else None
        if verbose == 2:
            print('Finished Training with updates:', updates)
        return hist

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given inputs.

        Args:
            x: Input tensor.

        Returns:
            Predicted output tensor.
        """
        return self(x)


class Model:
    """Abstract class for supported models."""

    sklearn_models = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'svr': SVR,
        'knn': KNeighborsRegressor,
        'decision_tree': DecisionTreeRegressor,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'xgboost': xgb.XGBRegressor
    }
    nn_models = {
        'mlp': MultiLayerPerceptron
    }
    available = {**sklearn_models, **nn_models}

    def __init__(self, **kwargs: Any) -> None:
        """Warn on direct instantiation.

        Args:
            **kwargs: Unused keyword arguments.
        """
        warnings.warn("This is an abstract class and should not be instantiated directly.", UserWarning)

    def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
        """Fit the model.

        Args:
            X: Input features.
            Y: Target values.
            **kwargs: Additional model-specific keyword arguments.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError
    
    def predict(self, X: Any) -> Any:
        """Predict output features.

        Args:
            X: Input features.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError


# Easy access to models
def get_model(model_name: str, **kwargs: Any) -> Any:
    """Construct a supported model instance.

    Args:
        model_name: Name of the model to construct.
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        A model instance or a ``MultiOutputRegressor`` wrapping a sklearn model.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name not in Model.available:
        raise ValueError(f"Model {model_name} not supported")
    if model_name in Model.nn_models or model_name == 'xgboost':
        return Model.available[model_name](**kwargs)
    else:
        return MultiOutputRegressor(Model.available[model_name](**kwargs))

# Get model parameters
def show_model_params(model_name: str) -> dict[str, Any]:
    """Return the default parameter dictionary for a supported model.

    Args:
        model_name: Name of the model to inspect.

    Returns:
        The parameter dictionary returned by ``get_params``.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name not in Model.available:
        raise ValueError(f"Model {model_name} not supported")
    if model_name in Model.nn_models or model_name == 'xgboost':
        model = Model.available[model_name]()
        return model.get_params()
    else:
        model = MultiOutputRegressor(Model.available[model_name]())
        return model.get_params()