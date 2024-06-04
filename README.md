# dl_assign_Democritos
This is a project that has been created as a task for the Deep Learning lesson of the Inter-Institutional MSc entitled "Artificial Intelligence" that is organized by The Department of Digital Systems, School of Informatics and Communication Technologies, of University of Piraeus, and the Institute of Informatics and Telecommunications of NCSR "Demokritos". url: "https://msc-ai.iit.demokritos.gr/en".

In this project various ML origin imputation methods are compared to Autencoders and Variational Autoencoders in the same dataset. The ML imputation methods that have been implemented are:

  •	Mean imputation (also serve as baseline)

  •	kNN Imputation, provided by the scikit learn python ML library

  •	Iterative Imputer, a new and experimental method of imputation also provided by scikit learn python library
  
  and have been implemented in “ML_imputers.py” code

Regarding the dl portion of this project, Autoencoders, Variational Autoencoders and Generative Adversarial Networks have been implemented I order to impute missing values. The different implementations are:

  •	Autoencoder with 1 LSTM layers for encoder and 1 LSTM layers for the decoder

  •	Autoencoder with 2 LSTM layers for encoder and 2 LSTM layers for the decoder

  •	VAE Autoencoder with 1 LSTM layers for encoder and 1 LSTM layers for the decoder

  •	VAE Autoencoder with 2 LSTM layers for encoder and 2 LSTM layers for the decoder

  •	GAN with 2 LSTM layers for generator and 2 LSTM layers for the discriminator

In all of the above configurations various hyperparameters were tested and as long as regularization technics like dropout layers and weight_decay.The python library used was pytorch.

The context of this project is data that origin from a ship’s Automatic Data Logging System (ADLM) system. The data concerns various operational parameters of the ship and is used by the crew to monitor its operational status. Since the original dataset provided didn’t have enough missing values, a random python function was used to create missing data in 4 different percentages (10%, 20%, 30% and 50%). This technic was chosen due to the fact that the actual mechanism of data missingness is likely to be caused by failure of a ship’s ADLM and therefore it should be considered as "Completely at Random" (MCAR).

The initial dataset has 228724 rows and 66 columns. In order to reduce the size of this repo, only the dataset that has been produced after the implementation of 10% missingness is provided, user can increase the missingness percentage using code provided.For the training validation and testing of the various Autoencoders, the dataset was splitted in train, validation and test datasets in the following percentages (70%, 10%, 20%). The dataset is constructed by concatenation of DIFFERENT ship trips that have different conditions (ship speed, load conditions) so it is considered ideal to test the “true” inference performance of the Autoencoders.

The code has been created with the use of python version 3.9.13. In order to recreate the same working enviroment (and to ensure trouble-free code execusion) it is advised to run under virtual enviroment that should be created with the use of requirements.txt (attached).

