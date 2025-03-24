from linear_variational import LinearReparameterization
from base_variational_layer import BaseVariationalLayer_
import torch
import torch.nn as nn

# LSTM layer that uses the reparameterization trick to sample weights from the posterior distribution
class LSTMReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,  # Input dimension
                 out_features,  # Output dimension (number of hidden units in the LSTM)
                 prior_mean=0,  # Mean of the prior distribution (Gaussian)
                 prior_variance=1,  # Variance of the prior distribution (Gaussian)
                 posterior_mu_init=0,  # Initial value for the posterior mean
                 posterior_rho_init=-3.0,  # Initial value for the posterior rho (determines variance via softplus)
                 bias=True):  # Whether the LSTM includes a bias term
        """
        Implements an LSTM layer with the reparameterization trick for variational inference.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_ to handle the base functionality.
        Uses LinearReparameterization for the LSTM gates (input-hidden and hidden-hidden connections).

        Parameters:
            prior_mean: float -> Mean of the prior distribution for KL divergence.
            prior_variance: float -> Variance of the prior distribution for KL divergence.
            posterior_mu_init: float -> Initial mean of the posterior.
            posterior_rho_init: float -> Initial rho of the posterior (transformed to std dev).
            in_features: int -> Dimensionality of input data.
            out_features: int -> Dimensionality of the output (hidden state).
            bias: bool -> Whether to include bias terms in the LSTM.
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

        # Input-hidden linear transformation (input to LSTM gates)
        self.ih = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=in_features,  # Input size
            out_features=out_features * 4,  # Output size is multiplied by 4 for LSTM gates (i, f, g, o)
            bias=bias)

        # Hidden-hidden linear transformation (hidden state to LSTM gates)
        self.hh = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=out_features,  # Hidden state size
            out_features=out_features * 4,  # Output size is multiplied by 4 for LSTM gates (i, f, g, o)
            bias=bias)

    # KL divergence loss calculation: combines KL losses from both input-hidden and hidden-hidden transformations
    def kl_loss(self):
        kl_i = self.ih.kl_loss()  # KL divergence for input-hidden weights
        kl_h = self.hh.kl_loss()  # KL divergence for hidden-hidden weights
        return kl_i + kl_h  # Total KL loss is the sum of both components

    # Forward pass for the LSTM
    def forward(self, X, hidden_states=None):
        """
        Performs the forward pass through the LSTM using reparameterization for weight sampling.
        
        Parameters:
            X: Input sequence of shape (batch_size, sequence_length, in_features).
            hidden_states: Tuple of (h_t, c_t) initial hidden and cell states. If None, initializes to zero.

        Returns:
            hidden_seq: Sequence of hidden states for each time step.
            (hidden_seq, c_ts): Tuple containing the final hidden and cell states.
            kl: KL divergence.
        """

        # Get batch size and sequence length
        batch_size, seq_size, _ = X.size()

        hidden_seq = []  # List to store hidden states over the sequence
        c_ts = []  # List to store cell states over the sequence

        # If hidden states (h_t, c_t) are not provided, initialize them to zeros
        if hidden_states is None:
            h_t, c_t = (torch.zeros(batch_size,
                                    self.out_features).to(X.device),
                        torch.zeros(batch_size,
                                    self.out_features).to(X.device))
        else:
            h_t, c_t = hidden_states  # Use the provided initial hidden and cell states

        HS = self.out_features  # Output size (number of hidden units)
        kl = 0  # Variable to accumulate KL divergence

        # Iterate over each time step in the sequence
        for t in range(seq_size):
            x_t = X[:, t, :]  # Get the input at time step t (batch_size, in_features)

            # Compute the input-hidden and hidden-hidden transformations using reparameterized weights
            ff_i, kl_i = self.ih(x_t)  # Input to LSTM gates, with corresponding KL loss
            ff_h, kl_h = self.hh(h_t)  # Hidden to LSTM gates, with corresponding KL loss
            gates = ff_i + ff_h  # Combine the two contributions to the LSTM gates

            kl += kl_i + kl_h  # Accumulate KL divergence from both transformations

            # Split the gates into their respective components (input, forget, cell, and output gates)
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # Input gate
                torch.sigmoid(gates[:, HS:HS * 2]),  # Forget gate
                torch.tanh(gates[:, HS * 2:HS * 3]),  # Cell gate (candidate state)
                torch.sigmoid(gates[:, HS * 3:]),  # Output gate
            )

            # Update the cell state
            c_t = f_t * c_t + i_t * g_t
            # Update the hidden state
            h_t = o_t * torch.tanh(c_t)

            # Store the hidden and cell states for this time step
            hidden_seq.append(h_t.unsqueeze(0))
            c_ts.append(c_t.unsqueeze(0))

        # Concatenate hidden and cell states over the sequence dimension
        hidden_seq = torch.cat(hidden_seq, dim=0)
        c_ts = torch.cat(c_ts, dim=0)
        
        # Reshape from (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_ts = c_ts.transpose(0, 1).contiguous()

        return hidden_seq, (hidden_seq, c_ts), kl
        