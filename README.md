# Variational Recurrent neural Networks (vRNN)

Uncertainty-aware power grid forecasting is essential for running a more resilient, cost-effective,
and cleaner power grid of integrated nuclear and renewable energy systems, where decisions are
made with a clearer view of potential outcomes. While Bayesian models offer a path to uncertainty
quantification, their high computational demands and poor scalability pose challenges for
real-time applications. This research introduces the development of variational recurrent neural networks
(vRNNs) as an effective solution for uncertainty-aware modeling with reduced computational
overhead, leveraging the power of variational inference. Unlike traditional networks with fixed parameters,
vRNNs assign probability distributions to parameters, allowing predictions to be sampled
and uncertainty quantified. The study applies four models including Gated Recurrent Units (GRU),
Long Short-Term Memory (LSTM), variational GRU (vGRU), and variational LSTM (vLSTM) to
minute-level power grid dataset, focusing on forecasting solar and wind power using weather conditions
like temperature, humidity and wind speed.

Each of the scripts to reproduce our results are available as Jupyter notebooks. Once each of scripts corresponding to the four models are run, the final plots can be obtained by running the notebooks inside scripts/results.

The dataset necessary to run these scripts is too large to store here, so it can be downloaded at: https://www.dropbox.com/scl/fi/exu3so0i1fghhg4dwp54m/PSML.csv?rlkey=r895e0nwbsc9c5jgaqiwmlpiw&st=8po4kxqb&dl=0
