import torch


def calculate_layer_entropy(hooked_model, layer_activations):
    """
    Performs the memory-intensive entropy calculation in a contained scope.
    """
    normalized_activations = hooked_model.ln_final(layer_activations)
    layer_logits = normalized_activations @ hooked_model.W_U
    stable_logits = layer_logits.to(torch.float32)
    entropy_per_position = torch.distributions.Categorical(logits=stable_logits).entropy()
    average_entropy = entropy_per_position.mean().item()
    
    # All large tensors (normalized_activations, layer_logits, etc.)
    # will be garbage collected when this function returns.
    return average_entropy