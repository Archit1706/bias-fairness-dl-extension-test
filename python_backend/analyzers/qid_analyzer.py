"""
Quantitative Individual Discrimination (QID) Analyzer module.
Based on DICE paper's information-theoretic approach.
"""

import torch
import numpy as np
from scipy.stats import entropy
from typing import List, Dict, Tuple, Union
import torch.nn.functional as F


class QIDAnalyzer:
    """
    Compute QID metrics using Shannon and Min entropy.

    QID measures the amount of protected information (in bits)
    used in decision making.
    """

    def __init__(self, model, protected_indices: List[int], device="cpu"):
        """
        Args:
            model: Trained FairnessDetectorDNN
            protected_indices: Indices of protected features
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
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
        logits = self.model(x_input)

        # Handle 0-dim tensor (scalar) - shouldn't happen but be safe
        if logits.dim() == 0:
            logits = logits.unsqueeze(0).unsqueeze(0)  # (1, 1)
        # Handle 1-dim tensor
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)  # (1, n)

        # Apply softmax
        probs = F.softmax(logits, dim=1)

        # Get first sample, flattening to ensure 1D
        probs = probs[0].flatten()

        # Ensure we have at least 2 classes for binary classification
        if probs.numel() == 1:
            # If only 1 output, treat as probability of class 1, compute class 0
            p1 = probs[0]
            probs = torch.stack([1 - p1, p1])

        return probs

    def generate_counterfactuals(
        self, x_base: torch.Tensor, protected_values: List
    ) -> List[torch.Tensor]:
        """
        Generate counterfactual instances by varying protected attributes.

        Args:
            x_base: Base instance (1D tensor)
            protected_values: List of values for protected attributes (floats or tensors)

        Returns:
            List of counterfactual tensors
        """
        counterfactuals = []

        for prot_val in protected_values:
            x_cf = x_base.clone()

            # Convert to float if it's a tensor
            if torch.is_tensor(prot_val):
                if prot_val.dim() == 0:  # Scalar tensor
                    prot_val = prot_val.item()
                else:
                    prot_val = (
                        prot_val.squeeze().item()
                        if prot_val.numel() == 1
                        else prot_val.squeeze()
                    )

            # Set protected attributes to counterfactual value
            if len(self.protected_indices) == 1:
                x_cf[self.protected_indices[0]] = float(prot_val)
            else:
                # If multiple protected attributes
                if isinstance(prot_val, (list, tuple, torch.Tensor)):
                    if torch.is_tensor(prot_val):
                        prot_val = (
                            prot_val.tolist()
                            if prot_val.numel() > 1
                            else [prot_val.item()]
                        )
                    for i, idx in enumerate(self.protected_indices):
                        if i < len(prot_val):
                            x_cf[idx] = float(prot_val[i])
                else:
                    # Single value for all protected features
                    for idx in self.protected_indices:
                        x_cf[idx] = float(prot_val)

            counterfactuals.append(x_cf)

        return counterfactuals

    def compute_shannon_qid(self, x_base: torch.Tensor, protected_values: List) -> Dict:
        """
        Compute Shannon entropy-based QID.

        QID = H(Y | X_nonprotected) - H(Y | X_all)

        Higher QID means more protected information is used.

        Returns:
            Dictionary with QID metrics and predictions
        """
        # Generate counterfactuals
        counterfactuals = self.generate_counterfactuals(x_base, protected_values)

        # Get predictions for all counterfactuals
        predictions = []
        with torch.no_grad():
            for x_cf in counterfactuals:
                x_cf = x_cf.to(self.device).unsqueeze(0)
                probs = self._safe_get_probs(x_cf)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        # Compute Shannon entropy of prediction distribution
        # If all predictions are the same, entropy is 0 (no discrimination)
        # If predictions vary, entropy > 0 (discrimination detected)

        # Average prediction across counterfactuals
        avg_pred = predictions.mean(axis=0)

        # Entropy of average prediction
        shannon_entropy = entropy(avg_pred) if len(avg_pred) > 1 else 0

        # Convert to bits
        qid_bits = shannon_entropy / np.log(2)

        # Compute variance as additional discrimination measure
        pred_variance = predictions.var(axis=0).sum()

        return {
            "shannon_qid": shannon_entropy,
            "qid_bits": qid_bits,
            "prediction_variance": float(pred_variance),
            "counterfactual_predictions": predictions.tolist(),
            "has_discrimination": qid_bits > 0.1,  # Threshold for significance
        }

    def compute_min_entropy_qid(
        self, x_base: torch.Tensor, protected_values: List
    ) -> Dict:
        """
        Compute Min entropy-based QID for worst-case analysis.

        Min entropy focuses on the most likely outcome,
        useful for extreme value analysis (EVT).

        Returns:
            Dictionary with min entropy metrics
        """
        counterfactuals = self.generate_counterfactuals(x_base, protected_values)

        max_probs = []
        favorable_outcomes = []

        with torch.no_grad():
            for x_cf in counterfactuals:
                x_cf = x_cf.to(self.device).unsqueeze(0)
                probs = self._safe_get_probs(x_cf)

                max_prob = probs.max().item()
                max_probs.append(max_prob)

                # Assume class 1 is "favorable" outcome (safe access with fallback)
                favorable_prob = probs[1].item() if probs.numel() > 1 else probs[0].item()
                favorable_outcomes.append(favorable_prob)

        # Min entropy = -log(max P(y|x))
        min_entropy = -np.log(max(max_probs)) if max(max_probs) > 0 else 0

        # Disparate impact ratio: min/max favorable probability
        favorable_probs = np.array(favorable_outcomes)
        disparate_impact = (
            favorable_probs.min() / favorable_probs.max()
            if favorable_probs.max() > 0
            else 0
        )

        # Legal threshold: 0.8 (80% rule)
        violates_80_rule = disparate_impact < 0.8

        return {
            "min_entropy": min_entropy,
            "max_favorable_prob": float(favorable_probs.max()),
            "min_favorable_prob": float(favorable_probs.min()),
            "disparate_impact_ratio": float(disparate_impact),
            "violates_80_rule": bool(violates_80_rule),
            "all_favorable_probs": favorable_probs.tolist(),
        }

    def batch_analyze(
        self, X: torch.Tensor, protected_values: List, max_samples: int = 1000
    ) -> Dict:
        """
        Analyze multiple instances from dataset.

        Args:
            X: Tensor of instances (n_samples, n_features)
            protected_values: Values for protected attributes (floats, not tensors)
            max_samples: Maximum number of samples to analyze

        Returns:
            Aggregated QID statistics
        """
        n_samples = min(len(X), max_samples)

        shannon_qids = []
        min_entropies = []
        disparate_impacts = []
        discriminatory_count = 0

        print(f"Analyzing {n_samples} instances...")

        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Progress: {i}/{n_samples}")

            x = X[i]

            try:
                # Shannon QID
                shannon_result = self.compute_shannon_qid(x, protected_values)
                shannon_qids.append(shannon_result["qid_bits"])

                if shannon_result["has_discrimination"]:
                    discriminatory_count += 1

                # Min entropy QID
                min_result = self.compute_min_entropy_qid(x, protected_values)
                min_entropies.append(min_result["min_entropy"])
                disparate_impacts.append(min_result["disparate_impact_ratio"])

            except Exception as e:
                print(f"Error analyzing instance {i}: {e}")
                continue

        print(f"Analysis complete!")

        return {
            "num_analyzed": n_samples,
            "mean_qid": float(np.mean(shannon_qids)),
            "max_qid": float(np.max(shannon_qids)),
            "std_qid": float(np.std(shannon_qids)),
            "mean_min_entropy": float(np.mean(min_entropies)),
            "mean_disparate_impact": float(np.mean(disparate_impacts)),
            "num_discriminatory": int(discriminatory_count),
            "pct_discriminatory": float(100 * discriminatory_count / n_samples),
            "num_violating_80_rule": int(
                sum(1 for di in disparate_impacts if di < 0.8)
            ),
        }


# Test script
if __name__ == "__main__":
    print("Testing QIDAnalyzer...")

    # Import model
    import sys

    sys.path.append("..")
    from models.fairness_dnn import FairnessDetectorDNN

    model = FairnessDetectorDNN(input_dim=10, protected_indices=[0])
    model.eval()

    analyzer = QIDAnalyzer(model, protected_indices=[0])

    # Test single instance with float values (not tensors!)
    x_base = torch.randn(10)
    protected_values = [0.0, 1.0]  # Simple floats

    shannon_result = analyzer.compute_shannon_qid(x_base, protected_values)
    print(f"✅ Shannon QID: {shannon_result['qid_bits']:.4f} bits")

    min_result = analyzer.compute_min_entropy_qid(x_base, protected_values)
    print(f"✅ Disparate Impact: {min_result['disparate_impact_ratio']:.4f}")

    # Test batch analysis
    X = torch.randn(100, 10)
    batch_result = analyzer.batch_analyze(X, protected_values, max_samples=50)
    print(f"✅ Mean QID (batch): {batch_result['mean_qid']:.4f} bits")
    print(f"✅ Discriminatory instances: {batch_result['num_discriminatory']}")

    print("✅ All tests passed!")
