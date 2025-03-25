# Development of Variational Neural Networks for Uncertainty Quantification of Nuclear Applications

<div style="text-align: justify"> Uncertainty-aware artificial intelligence (AI) and machine learning (ML) are essential for the safe
and reliable deployment of neural-based methods in safety-critical engineering applications. The
field of nuclear energy exhibits especially strict licensing and regulation, necessitating the development
and testing of uncertainty-aware models before the industry is able to realize the full value of
recent advances in AI/ML. While Bayesian models offer a path to uncertainty quantification, their
high computational demands and poor scalability pose challenges for large datasets and real-time
applications. This research introduces the development of variational neural networks (VNN) as
an effective solution for uncertainty-aware modeling with reduced computational overhead, leveraging
the power of variational inference. Unlike traditional networks with fixed parameters, VNNs
assign probability distributions to parameters, allowing predictions to be sampled and uncertainty
quantified. This study leverages previous progress in the development of VNNs and extends their
abilities to predict the neutron distribution in the HOLOS-Quad 22MWt high-temperature gas-cooled
microreactor using the angles of the 8 cylindrical control drums.</div>

Further, we introduce the development and deployment of variational recurrent neural networks
(VRNN) as a promising path towards integrating sensor data into AI/ML models for time-series
forecasting of future system states and associated uncertainties. We use a custom loss function in
variational models to integrate mean squared error with the Kullback-Leibler divergence, guiding
parameters toward meaningful posterior distributions. It is found that variational models achieve
competitive performance with deterministic counterparts while delivering uncertainty estimates.
This work demonstrates that variational inference provides a scalable, computationally efficient
framework for real-time digital twins, with plans to apply the methodology to nuclear plant data in
the near future.

Keywords: Variational Inference, Feedforward Neural Networks, Recurrent Neural Networks,
HTGR, Integrated Energy Systems

Each of the scripts to reproduce our results are available as Jupyter notebooks. Once each of scripts corresponding to the four models are run, the final plots can be obtained by running the notebooks inside scripts/results. The requirements.txt file is avaialable in /scripts to reconstruct the venv.

The dataset necessary to run these scripts is too large to store here, so it can be downloaded at: https://www.dropbox.com/scl/fi/exu3so0i1fghhg4dwp54m/PSML.csv?rlkey=r895e0nwbsc9c5jgaqiwmlpiw&st=8po4kxqb&dl=0

Once the dataset is downloaded, store it as /scripts/dataset/PSML.csv to allow proper loading within the notebooks.
