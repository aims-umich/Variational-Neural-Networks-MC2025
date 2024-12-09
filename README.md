# NERS590-vRNN

## Abstract
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
