import torch.nn.functional as F


def weighted_loss_function(
    reconstruction,
    original_inputs,
    weights
):
    league_loss = F.mse_loss(reconstruction[:, :weights["league_dim"]], original_inputs[:, :weights["league_dim"]])

    national_loss = F.mse_loss(
        reconstruction[:, weights["league_dim"]:weights["league_dim"] + weights["national_dim"]],
        original_inputs[:, weights["league_dim"]:weights["league_dim"] + weights["national_dim"]]
    )

    league_history_loss = F.mse_loss(
        reconstruction[:, weights["league_dim"] + weights["national_dim"]:weights["league_dim"] + weights["national_dim"] + weights["league_history_dim"]],
        original_inputs[:, weights["league_dim"] + weights["national_dim"]:weights["league_dim"] + weights["national_dim"] + weights["league_history_dim"]]
    )

    attributes_loss = F.mse_loss(
        reconstruction[:, weights["league_dim"] + weights["national_dim"] + weights["league_history_dim"]:-weights["categorical_dim"]],
        original_inputs[:, weights["league_dim"] + weights["national_dim"] + weights["league_history_dim"]:-weights["categorical_dim"]]
    )

    categorical_loss = F.mse_loss(
        reconstruction[:, -weights["categorical_dim"]:],
        original_inputs[:, -weights["categorical_dim"]:]
    )

    reconstruction_loss = (
        weights["league_reconstruction"] * league_loss +
        weights["national_reconstruction"] * national_loss +
        weights["league_history_reconstruction"] * league_history_loss +
        weights["attributes_reconstruction"] * attributes_loss +
        weights["categorical_reconstruction"] * categorical_loss
    )

    return reconstruction_loss
