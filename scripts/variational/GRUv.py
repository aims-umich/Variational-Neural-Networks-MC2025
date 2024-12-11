from linear_variational import LinearReparameterization
from base_variational_layer import BaseVariationalLayer_
import torch
import torch.nn as nn


# GRU layer that uses the reparameterization trick to sample weights from the posterior distribution
class GRUReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,  # Input dimension (size of the input feature vector)
                 out_features,  # Output dimension (number of hidden units in the GRU)
                 prior_mean=0,  # Mean of the prior distribution (Gaussian)
                 prior_variance=1,  # Variance of the prior distribution (Gaussian)
                 posterior_mu_init=0,  # Initial value for the posterior mean
                 posterior_rho_init=-3.0,  # Initial value for the posterior rho (determines variance via softplus)
                 bias=True):  # Whether the GRU includes a bias term
        """
        Implements a GRU layer with the reparameterization trick for variational inference.
        This GRU layer uses linear transformations with weight sampling from posterior distributions.

        Inherits from BaseVariationalLayer_ to handle base functionality.
        Uses LinearReparameterization for the GRU gates (input-hidden and hidden-hidden connections).

        Parameters:
            prior_mean: float -> Mean of the prior distribution for KL divergence.
            prior_variance: float -> Variance of the prior distribution for KL divergence.
            posterior_mu_init: float -> Initial mean of the posterior.
            posterior_rho_init: float -> Initial rho of the posterior (transformed to std dev).
            in_features: int -> Dimensionality of input data.
            out_features: int -> Dimensionality of the output (hidden state).
            bias: bool -> Whether to include bias terms in the GRU.
        """
        super().__init__()  # Initialize the base class

        # Initialize layer parameters
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init  # Posterior mean for reparameterization
        self.posterior_rho_init = posterior_rho_init  # Posterior rho for reparameterization (std dev through softplus)
        self.bias = bias

        # Input-hidden linear transformation (input to GRU gates)
        # GRU has 3 gates: update (z), reset (r), and new (n) gates
        self.ih = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=in_features,  # Input size
            out_features=out_features * 3,  # Output size is multiplied by 3 for GRU gates (z, r, n)
            bias=bias)

        # Hidden-hidden linear transformation (hidden state to GRU gates)
        self.hh = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=out_features,  # Hidden state size
            out_features=out_features * 3,  # Output size is multiplied by 3 for GRU gates (z, r, n)
            bias=bias)

    # KL divergence loss calculation: combines KL losses from both input-hidden and hidden-hidden transformations
    def kl_loss(self):
        """
        Calculates the total KL divergence loss for the input-hidden and hidden-hidden transformations.
        """
        kl_i = self.ih.kl_loss()  # KL divergence for input-hidden weights
        kl_h = self.hh.kl_loss()  # KL divergence for hidden-hidden weights
        return kl_i + kl_h  # Total KL loss is the sum of both components

    # Forward pass for the GRU
    def forward(self, X, hidden_states=None):
        """
        Performs the forward pass through the GRU using reparameterization for weight sampling.
        
        Parameters:
            X: Input sequence of shape (batch_size, sequence_length, in_features).
            hidden_states: Tensor of initial hidden states. If None, initializes to zero.

        Returns:
            hidden_seq: Sequence of hidden states for each time step.
            hidden_last: Final hidden state.
            kl: KL divergence.
        """

        # Get batch size and sequence length
        batch_size, seq_size, _ = X.size()

        hidden_seq = []  # List to store hidden states over the sequence

        # If hidden states are not provided, initialize them to zeros
        if hidden_states is None:
            h_t = torch.zeros(batch_size, self.out_features).to(X.device)  # Initialize hidden state
        else:
            h_t = hidden_states  # Use the provided initial hidden state

        HS = self.out_features  # Output size (number of hidden units)
        kl = 0  # Variable to accumulate KL divergence

        # Iterate over each time step in the sequence
        for t in range(seq_size):
            x_t = X[:, t, :]  # Get the input at time step t (batch_size, in_features)

            # Compute the input-hidden and hidden-hidden transformations using reparameterized weights
            ff_i, kl_i = self.ih(x_t)  # Input to GRU gates, with corresponding KL loss
            ff_h, kl_h = self.hh(h_t)  # Hidden to GRU gates, with corresponding KL loss
            gates = ff_i + ff_h  # Combine the two contributions to the GRU gates

            kl += kl_i + kl_h  # Accumulate KL divergence from both transformations

            # Split the gates into their respective components (update, reset, and new gates)
            z_t, r_t, n_t = (
                torch.sigmoid(gates[:, :HS]),  # Update gate (z_t)
                torch.sigmoid(gates[:, HS:HS * 2]),  # Reset gate (r_t)
                torch.tanh(gates[:, HS * 2:]),  # New gate (n_t) to generate candidate hidden state
            )

            # Compute the next hidden state
            h_t = (1 - z_t) * n_t + z_t * h_t  # Combine the new candidate and previous hidden state

            # Store the hidden state for this time step
            hidden_seq.append(h_t.unsqueeze(0))

        # Concatenate hidden states over the sequence dimension
        hidden_seq = torch.cat(hidden_seq, dim=0)

        # Reshape from (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        # Return the hidden sequence, final hidden states, and KL loss
        return hidden_seq, h_t, kl
        