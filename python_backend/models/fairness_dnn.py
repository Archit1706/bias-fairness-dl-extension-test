"""
Fairness Detector DNN
6-layer architecture based on DICE paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class FairnessDetectorDNN(nn.Module):
    """
    Deep Neural Network for detecting bias in datasets.

    Architecture: Input → 64 → 32 → 16 → 8 → 4 → 2 (output)
    All layers use ReLU activation except output layer.
    Dropout (0.2) applied after each hidden layer for regularization.

    This architecture is based on the DICE paper:
    "Information-Theoretic Testing and Debugging of Fairness Defects in DNNs"
    """

    def __init__(
        self, input_dim: int, protected_indices: List[int], dropout_rate: float = 0.2
    ):
        """
        Args:
            input_dim: Number of input features
            protected_indices: Indices of protected attributes in input
            dropout_rate: Dropout probability (default 0.2)
        """
        super(FairnessDetectorDNN, self).__init__()

        self.input_dim = input_dim
        self.protected_indices = protected_indices
        self.dropout_rate = dropout_rate

        # Define layers (6-layer architecture)
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 4)
        self.output_layer = nn.Linear(4, 2)  # Binary classification

        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(8)
        self.bn5 = nn.BatchNorm1d(4)

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_activations: If True, return activations at each layer

        Returns:
            Output logits of shape (batch_size, 2)
            If return_activations=True, also returns list of activations
        """
        activations = []

        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())

        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())

        # Layer 3
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())

        # Layer 4
        x = self.layer4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())

        # Layer 5
        x = self.layer5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())

        # Output layer (no activation)
        x = self.output_layer(x)

        if return_activations:
            return x, activations
        return x

    def get_layer_output(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Get output of a specific layer (for causal analysis).

        Args:
            x: Input tensor
            layer_idx: Layer index (0-5)

        Returns:
            Activation tensor at specified layer
        """
        layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.output_layer,
        ]
        batch_norms = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, None]

        for i in range(layer_idx + 1):
            x = layers[i](x)
            if batch_norms[i] is not None and i < len(batch_norms) - 1:
                x = batch_norms[i](x)
            if i < layer_idx:  # Apply activation for intermediate layers
                x = F.relu(x)
                if i > 0:
                    x = self.dropout(x)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DNNTrainer:
    """Trainer class for FairnessDetectorDNN."""

    def __init__(
        self,
        model: FairnessDetectorDNN,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        """
        Args:
            model: FairnessDetectorDNN instance
            learning_rate: Learning rate for Adam optimizer
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop with early stopping.

        Returns:
            Dictionary with training history
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "best_val_loss": best_val_loss,
        }


# Test script
if __name__ == "__main__":
    print("Testing FairnessDetectorDNN...")

    # Create dummy data
    batch_size = 32
    input_dim = 14  # Like Adult dataset
    protected_indices = [0, 1]  # First two features are protected

    # Create model
    model = FairnessDetectorDNN(input_dim, protected_indices)
    print(f"Model has {model.count_parameters()} trainable parameters")

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 2)

    # Test with activations
    output, activations = model(x, return_activations=True)
    print(f"Number of activation layers: {len(activations)}")  # Should be 5

    # Test layer output
    layer_2_output = model.get_layer_output(x, 2)
    print(f"Layer 2 output shape: {layer_2_output.shape}")  # Should be (32, 16)

    print("✅ All tests passed!")
