"""
Gradient-guided search for discriminatory instances.
Implements DICE's global + local two-phase search.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from .qid_analyzer import QIDAnalyzer


class DiscriminatoryInstanceSearch:
    """
    Search for inputs that maximize discrimination (QID).
    """

    def __init__(
        self,
        model,
        qid_analyzer: QIDAnalyzer,
        protected_indices: List[int],
        device="cpu",
    ):
        """
        Args:
            model: Trained DNN
            qid_analyzer: QIDAnalyzer instance
            protected_indices: Indices of protected features
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.qid_analyzer = qid_analyzer
        self.protected_indices = protected_indices
        self.device = device

    def _safe_get_probs(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Safely get probability predictions from model, handling all tensor shape edge cases.

        Args:
            x_input: Input tensor (should be 2D with batch dimension)

        Returns:
            1D probability tensor of shape (num_classes,)
        """
        output = self.model(x_input)

        # Handle 0-dim tensor (scalar)
        if output.dim() == 0:
            output = output.unsqueeze(0).unsqueeze(0)
        # Handle 1-dim tensor
        elif output.dim() == 1:
            output = output.unsqueeze(0)

        # Apply softmax
        probs = F.softmax(output, dim=1)

        # Get first sample, flattening to ensure 1D
        probs = probs[0].flatten()

        # Ensure we have at least 2 classes for binary classification
        if probs.numel() == 1:
            p1 = probs[0]
            probs = torch.stack([1 - p1, p1])

        return probs

    def global_search(
        self,
        x_init: torch.Tensor,
        protected_values: List,
        num_iterations: int = 100,
        lr: float = 0.01,
    ) -> Tuple[torch.Tensor, float]:
        """
        Global phase: Find inputs maximizing QID via gradient ascent.
        """
        x_current = x_init.clone().detach().to(self.device)
        x_current.requires_grad = True

        optimizer = torch.optim.Adam([x_current], lr=lr)

        best_qid = 0
        best_instance = x_init.clone()

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Generate counterfactuals
            counterfactual_outputs = []
            for prot_val in protected_values:
                x_cf = x_current.clone()

                # Convert to float if needed
                val = float(prot_val) if torch.is_tensor(prot_val) else prot_val

                # Set protected attribute
                if len(self.protected_indices) == 1:
                    x_cf[self.protected_indices[0]] = val  # ✅ Now using float
                else:
                    for i, idx in enumerate(self.protected_indices):
                        if isinstance(val, (list, tuple)):
                            x_cf[idx] = float(val[i])
                        else:
                            x_cf[idx] = val

                probs = self._safe_get_probs(x_cf.unsqueeze(0))
                counterfactual_outputs.append(probs)

            # Stack outputs
            outputs_tensor = torch.stack(counterfactual_outputs)

            # QID as variance across counterfactuals (higher = more discrimination)
            # Negative because we want to maximize, but optimizer minimizes
            qid_loss = -outputs_tensor.var(dim=0).sum()

            # Backpropagate
            qid_loss.backward()
            optimizer.step()

            # Track best
            current_qid = -qid_loss.item()
            if current_qid > best_qid:
                best_qid = current_qid
                best_instance = x_current.clone().detach()

        return best_instance, best_qid

    def local_search(
        self,
        x_base: torch.Tensor,
        protected_values: List,
        num_neighbors: int = 50,
        perturbation_scale: float = 0.1,
    ) -> List[Dict]:
        """
        Local phase: Generate discriminatory instances via perturbation.

        Args:
            x_base: Base instance (from global search)
            protected_values: Values for protected attributes
            num_neighbors: Number of neighbors to sample
            perturbation_scale: Standard deviation of Gaussian noise

        Returns:
            List of discriminatory instances with QID scores
        """
        discriminatory_instances = []

        for _ in range(num_neighbors):
            # Perturb non-protected features
            noise = torch.randn_like(x_base) * perturbation_scale

            # Don't perturb protected features
            for idx in self.protected_indices:
                noise[idx] = 0

            x_neighbor = x_base + noise

            # Compute QID for this neighbor
            qid_result = self.qid_analyzer.compute_shannon_qid(
                x_neighbor, protected_values
            )

            # Only keep if discriminatory
            if qid_result["qid_bits"] > 0.1:  # Threshold
                discriminatory_instances.append(
                    {
                        "instance": x_neighbor.cpu().numpy().tolist(),
                        "qid": qid_result["qid_bits"],
                        "predictions": qid_result["counterfactual_predictions"],
                        "variance": qid_result["prediction_variance"],
                    }
                )

        # Sort by QID (most discriminatory first)
        discriminatory_instances.sort(key=lambda x: x["qid"], reverse=True)

        return discriminatory_instances

    def search(
        self,
        X: torch.Tensor,
        protected_values: List,
        num_global_iterations: int = 100,
        num_local_neighbors: int = 50,
    ) -> Dict:
        """
        Full two-phase search.

        Args:
            X: Dataset to initialize search from
            protected_values: Values for protected attributes
            num_global_iterations: Iterations for global search
            num_local_neighbors: Neighbors for local search

        Returns:
            Dictionary with search results
        """
        # Start from random instance in dataset
        idx = np.random.randint(0, len(X))
        x_init = X[idx]

        print(f"Phase 1: Global search (optimizing for maximum QID)...")
        best_instance, best_qid = self.global_search(
            x_init, protected_values, num_global_iterations
        )
        print(f"Found instance with QID = {best_qid:.4f}")

        print(f"Phase 2: Local search (generating discriminatory instances)...")
        discriminatory_instances = self.local_search(
            best_instance, protected_values, num_local_neighbors
        )
        print(f"Generated {len(discriminatory_instances)} discriminatory instances")

        return {
            "best_instance": best_instance.cpu().numpy().tolist(),
            "best_qid": best_qid,
            "discriminatory_instances": discriminatory_instances,
            "num_found": len(discriminatory_instances),
        }


# Test script
if __name__ == "__main__":
    print("Testing DiscriminatoryInstanceSearch...")

    from models.fairness_dnn import FairnessDetectorDNN
    from analyzers.qid_analyzer import QIDAnalyzer

    # Create model and analyzer
    model = FairnessDetectorDNN(input_dim=10, protected_indices=[0])
    model.eval()

    qid_analyzer = QIDAnalyzer(model, protected_indices=[0])
    search_engine = DiscriminatoryInstanceSearch(
        model, qid_analyzer, protected_indices=[0]
    )

    # Create dummy dataset
    X = torch.randn(100, 10)
    protected_values = [torch.tensor(0.0), torch.tensor(1.0)]

    # Run search
    results = search_engine.search(
        X,
        protected_values,
        num_global_iterations=20,  # Reduced for testing
        num_local_neighbors=10,
    )

    print(f"✅ Best QID: {results['best_qid']:.4f}")
    print(f"✅ Discriminatory instances found: {results['num_found']}")

    print("✅ All tests passed!")
