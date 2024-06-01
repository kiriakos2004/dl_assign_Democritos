# dl_assign_Democritos
This is a project that has been created as a task for the Deep Learning lesson of the Inter-Institutional MSc entitled "Artificial Intelligence" that is organized by The Department of Digital Systems, School of Informatics and Communication Technologies, of University of Piraeus, and the Institute of Informatics and Telecommunications of NCSR "Demokritos". url: "https://msc-ai.iit.demokritos.gr/en".

In this project various ML origin imputation methods are compared to Autencoders and Variational Autoencoders in the same dataset. The ML imputation methods that have been implemented are:

•	Mean imputation (also serve as baseline)

•	kNN Imputation, provided by the scikit learn python ML library

•	Iterative Imputer, a new and experimental method of imputation also provided by scikit learn python library

Regarding the dl portion of this project, Autoencoders and Variational Autoencoders have been implemented I order to impute missing values. The different implementations are:

•	Autoencoder with 1 LSTM layers for encoder and 1 LSTM layers for the decoder

•	Autoencoder with 2 LSTM layers for encoder and 2 LSTM layers for the decoder

•	VAE Autoencoder with 1 LSTM layers for encoder and 1 LSTM layers for the decoder

•	VAE Autoencoder with 2 LSTM layers for encoder and 2 LSTM layers for the decoder

In all of the above configurations various hyperparameters were tested and as long as regularization technics like dropout layers and weight_decay.

