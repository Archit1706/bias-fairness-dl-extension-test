"""
Causal layer and neuron localization for bias debugging.
Based on DICE Algorithm 2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


class CausalDebugger:
    """Localize biased layers and neurons using causal interventions."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Build layers list from model
        self.layers = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.layer5,
            self.model.output_layer
        ]

    def localize_biased_layer(
        self, discriminatory_instances: List[Dict], accuracy_threshold: float = 0.05
    ) -> Dict:
        """
        Identify which layer contributes most to bias.

        Uses gradient-based sensitivity analysis.
        """
        num_layers = len(self.layers)
        layer_sensitivities = []

        for layer_idx in range(num_layers):
            total_sensitivity = 0

            # Analyze top 20 discriminatory instances
            for instance_data in discriminatory_instances[:20]:
                x = torch.tensor(instance_data["instance"], dtype=torch.float32).to(
                    self.device
                )
                x.requires_grad_(True)

                # Get activations at this layer
                activations = self.model.get_layer_output(x.unsqueeze(0), layer_idx)
                # Ensure activations has proper shape (batch_size, features)
                # Handle 0-dim (scalar), 1-dim, and 2-dim tensors
                while activations.dim() < 2:
                    activations = activations.unsqueeze(0)

                # Compute gradient of activations w.r.t. input
                grad_outputs = torch.ones_like(activations)
                gradients = torch.autograd.grad(
                    outputs=activations,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    retain_graph=False,
                )[0]

                # Sensitivity = magnitude of gradients
                sensitivity = gradients.abs().mean().item()
                total_sensitivity += sensitivity

            avg_sensitivity = total_sensitivity / min(20, len(discriminatory_instances))

            layer_sensitivities.append(
                {
                    "layer_idx": layer_idx,
                    "layer_name": f"Layer {layer_idx + 1}",
                    "sensitivity": avg_sensitivity,
                    "neuron_count": self.layers[layer_idx].out_features,
                }
            )

        # Find layer with highest sensitivity
        most_biased_layer = max(layer_sensitivities, key=lambda x: x["sensitivity"])

        return {"biased_layer": most_biased_layer, "all_layers": layer_sensitivities}

    def localize_biased_neurons(
        self, layer_idx: int, discriminatory_instances: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """
        Identify specific neurons encoding protected information.
        """
        layer = self.layers[layer_idx]
        num_neurons = layer.out_features

        neuron_impacts = np.zeros(num_neurons)

        # Analyze each discriminatory instance
        for instance_data in discriminatory_instances[:30]:
            x = torch.tensor(instance_data["instance"], dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                # Get activations at this layer
                activations = self.model.get_layer_output(x.unsqueeze(0), layer_idx)
                # Ensure activations has batch dimension before indexing
                # Handle 0-dim (scalar), 1-dim, and 2-dim tensors
                while activations.dim() < 2:
                    activations = activations.unsqueeze(0)
                activations = activations[0]

                # Each neuron's activation magnitude = its impact
                neuron_impacts += activations.abs().cpu().numpy()

        # Average across instances
        neuron_impacts /= min(30, len(discriminatory_instances))

        # Get top-k neurons
        top_neuron_indices = np.argsort(neuron_impacts)[-top_k:][::-1]

        return [
            {
                "neuron_idx": int(idx),
                "impact_score": float(neuron_impacts[idx]),
                "layer_idx": layer_idx,
            }
            for idx in top_neuron_indices
        ]
