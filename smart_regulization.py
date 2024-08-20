import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# lambda_smoothness = 0.1  # Adjust based on experimentation
# beta = 0.5  # Adjust based on experimentation

def smart_loss(model, embeddings_1, embeddings_2, labels, epsilon=1e-5, alpha=1e-3, K_steps=3, lambda_adv=0.5):
    """
    SMART loss implementation with MSE as task loss and KL divergence for adversarial regularization.
    """
    # Step 1: Compute task loss (MSE in this case)
    logits = model.predict_similarity_SMART(embeddings_1, embeddings_2)
    task_loss = F.mse_loss(logits, labels.float())

    # Step 2: Initialize perturbation delta for both input embeddings
    delta_1 = torch.zeros_like(embeddings_1).uniform_(-epsilon, epsilon)
    delta_2 = torch.zeros_like(embeddings_2).uniform_(-epsilon, epsilon)

    # Step 3: Adversarial perturbation generation (PGD)
    for _ in range(K_steps):
        delta_1.requires_grad_()
        delta_2.requires_grad_()

        # Add perturbations
        perturbed_embeddings_1 = embeddings_1 + delta_1
        perturbed_embeddings_2 = embeddings_2 + delta_2

        # Get logits for perturbed embeddings
        adv_logits = model.predict_similarity_SMART(perturbed_embeddings_1, perturbed_embeddings_2)
        
        # Compute adversarial loss (same as task loss: MSE)
        adv_loss = F.mse_loss(adv_logits, labels.float())

        # Get gradients of loss w.r.t perturbation
        # Use retain_graph=True only for the gradient calculation of perturbations
        adv_grad_1 = torch.autograd.grad(adv_loss, delta_1, retain_graph=True)[0]
        adv_grad_2 = torch.autograd.grad(adv_loss, delta_2, retain_graph=True)[0]

        # Update perturbations using gradient ascent
        delta_1 = delta_1 + alpha * adv_grad_1.sign()
        delta_2 = delta_2 + alpha * adv_grad_2.sign()

        # Clip perturbations to ensure they are small
        delta_1 = torch.clamp(delta_1, -epsilon, epsilon)
        delta_2 = torch.clamp(delta_2, -epsilon, epsilon)

    # Step 4: Compute adversarial regularization loss
    perturbed_embeddings_1 = embeddings_1 + delta_1
    perturbed_embeddings_2 = embeddings_2 + delta_2
    adv_logits = model.predict_similarity_SMART(perturbed_embeddings_1, perturbed_embeddings_2)
    
    # Compute KL divergence loss between original and adversarial logits
    adv_loss = F.kl_div(F.log_softmax(logits, dim=-1),
                        F.softmax(adv_logits, dim=-1), reduction='batchmean')

    # Step 5: Combine task loss and adversarial loss
    final_loss = task_loss + lambda_adv * adv_loss

    return final_loss