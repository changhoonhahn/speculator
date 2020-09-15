''' 

script to train speculator for DESI 


'''
import os 
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
# --- speculator --- 
from speculator import SpectrumPCA
from speculator import Speculator

dtype = tf.float32


dir_gqp = '/Users/ChangHoon/data/gqp_mc/'


def DESI_simpledust(log=False): 
    ''' speculator for DESI with simple dust modle. In this case, 

    theta = [b1, b2, b3, b4, g1, g2, tau, tage]

    '''
    # training set 
    fwave   = os.path.join(dir_gqp, 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    fthetas = [os.path.join(dir_gqp, 'speculator',
        'DESI_simpledust.theta_train.%i.npy' % i) for i in range(10)]
    fspecs  = [os.path.join(dir_gqp, 'speculator', 
        'DESI_simpledust.logspectrum_fsps_train.%i.npy' % i) for i in range(10)]

    # theta = [b1, b2, b3, b4, g1, g2, tau, tage]
    n_param = 8 
    n_wave  = len(wave)
    n_pcas  = 20 
    
    # train PCA basis 
    print('training PCA bases') 
    PCABasis = SpectrumPCA(
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pcas,              # number of pca coefficients to include in the basis 
            spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
            parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)

    print('  compute shift and scale') 
    PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
    print('  train PCA') 
    PCABasis.train_pca()
    print('  transform and stack training data') 
    PCABasis.transform_and_stack_training_data(
            os.path.join(dir_gqp, 'speculator', 'DESI_simpledust'), 
            retain=True) 
    
    # train Speculator 
    training_theta  = tf.convert_to_tensor(PCABasis.training_parameters.astype(np.float32))
    training_pca    = tf.convert_to_tensor(PCABasis.training_pca.astype(np.float32))

    speculator = Speculator(
            n_parameters=n_param, # number of model parameters 
            wavelengths=wave, # array of wavelengths
            pca_transform_matrix=PCABasis.pca_transform_matrix,
            parameters_shift=PCABasis.parameters_shift, 
            parameters_scale=PCABasis.parameters_scale, 
            pca_shift=PCABasis.pca_shift, 
            pca_scale=PCABasis.pca_scale, 
            spectrum_shift=PCABasis.spectrum_shift, 
            spectrum_scale=PCABasis.spectrum_scale, 
            n_hidden=[256, 256, 256], # network architecture (list of hidden units per layer)
            restore=False, 
            optimizer=tf.keras.optimizers.Adam()) # optimizer for model training

    # cooling schedule
    validation_split = 0.1
    lr = [1e-3, 1e-4, 1e-5, 1e-6]
    batch_size = [1000, 10000, 50000, int((1-validation_split) * training_theta.shape[0])]
    gradient_accumulation_steps = [1, 1, 1, 10] # split the largest batch size into 10 when computing gradients to avoid memory overflow

    # early stopping set up
    patience = 20

    # train using cooling/heating schedule for lr/batch-size
    for i in range(len(lr)):
        print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))
        # set learning rate
        speculator.optimizer.lr = lr[i]

        # split into validation and training sub-sets
        n_validation = int(training_theta.shape[0] * validation_split)
        n_training = training_theta.shape[0] - n_validation
        training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

        # create iterable dataset (given batch size)
        training_data = tf.data.Dataset.from_tensor_slices(
                (training_theta[training_selection], training_pca[training_selection])).shuffle(n_training).batch(batch_size[i])

        # set up training loss
        training_loss   = [np.infty]
        validation_loss = [np.infty]
        best_loss       = np.infty
        early_stopping_counter = 0

        # loop over epochs
        while early_stopping_counter < patience:

            # loop over batches
            for theta, pca in training_data:

                # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                if gradient_accumulation_steps[i] == 1:
                    loss = speculator.training_step(theta, pca)
                else:
                    loss = speculator.training_step_with_accumulated_gradients(theta, pca, accumulation_steps=gradient_accumulation_steps[i])

            # compute validation loss at the end of the epoch
            validation_loss.append(speculator.compute_loss(training_theta[~training_selection], training_pca[~training_selection]).numpy())

            # early stopping condition
            if validation_loss[-1] < best_loss:
                best_loss = validation_loss[-1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                speculator.update_emulator_parameters()
                speculator.save(os.path.join(dir_gqp, 'speculator', '_DESI_simpledust_model%s' % ['', '.log'][log]))

                attributes = list([
                        list(speculator.W_), 
                        list(speculator.b_), 
                        list(speculator.alphas_), 
                        list(speculator.betas_), 
                        speculator.pca_transform_matrix_,
                        speculator.pca_shift_,
                        speculator.pca_scale_,
                        speculator.spectrum_shift_,
                        speculator.spectrum_scale_,
                        speculator.parameters_shift_, 
                        speculator.parameters_scale_,
                        speculator.wavelengths])

                # save attributes to file
                f = open(os.path.join(dir_gqp, 'speculator',
                    'DESI_simpledust_model%s.pkl' % ['', '.log'][log]), 'wb')
                pickle.dump(attributes, f)
                f.close()
                print('Validation loss = %s' % str(best_loss))

    speculator = Speculator(restore=True, 
            restore_filename=os.path.join(dir_gqp, 'speculator', '_DESI_simpledust_model%s' % ['', '.log'][log]))
    
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for i, _theta in enumerate(training_theta[~training_selection][:5]): 
        log_spec = speculator.log_spectrum_(_theta) # compute log spectrum

        sub.plot(speculator.wavelengths, log_spec, c='C%i' % i, ls='-') 
        print(log_spec)
        sub.plot(wave, PCABasis.training_spectrum[~training_selection][i], c='C%i' % i, ls=':') 
        print(PCABasis.training_spectrum[~training_selection][i])
    sub.set_xlim(3e3, 1e4)
    fig.savefig('_DESI_simpledust_model.png', bbox_inches='tight') 
    return None 


if __name__=="__main__":
    DESI_simpledust(log=True)
