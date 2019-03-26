import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from scipy.misc import imresize

from storage_utils import save_statistics

from utils import create_labels_mult_decoder
from utils import create_labels_holes

import matplotlib.pyplot as plt


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, gpu_id, continue_from_epoch=-1, device=None, loss_weights=[0.333, 0.333, 0.334], model_arc='standard', input_size=128, hole_context=0, loss_multiplier=False):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        # if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
        #     if "," in gpu_id:
        #         self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
        #     else:
        #         self.device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

        #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
        #     print("use GPU")
        #     print("GPU ID {}".format(gpu_id))
        # else:
        #     print("use CPU")
        #     self.device = torch.device('cpu')  # sets the device to be CPU

        self.device = device
        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        if torch.cuda.device_count()>1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            # self.device = self.device[0]
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 0.

        self.loss_weights = loss_weights
        self.model_arc = model_arc
        self.input_size = input_size
        self.hole_context = hole_context
        self.loss_multiplier = loss_multiplier

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs

        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()


    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x):


        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if self.model_arc == 'multdec':

            y1, y2, y3 = create_labels_mult_decoder(x) 
            x  = torch.tensor(x).float().to(device=self.device)
            y1 = torch.tensor(y1).float().to(device=self.device)
            y2 = torch.tensor(y2).float().to(device=self.device)
            y3 = torch.tensor(y3).float().to(device=self.device)

            out1, out2, out3 = self.model.forward(x)

            loss_1 = F.mse_loss(input=out1, target=y1)
            loss_2 = F.mse_loss(input=out2, target=y2)
            loss_3 = F.mse_loss(input=out3, target=y3)
            loss = self.loss_weights[0]*loss_1 + self.loss_weights[1]*loss_2 + self.loss_weights[2]*loss_3

        elif self.model_arc == 'holes':

            new_x, masks = create_labels_holes(x, self.hole_context)
            y = np.array(x, copy=True)

            new_x = torch.tensor(new_x).float().to(device=self.device)
            y = torch.tensor(y).float().to(device=self.device)
            masks = torch.tensor(masks).float().to(device=self.device)

            out = self.model.forward(new_x)
            out_mask = torch.mul(out, masks)
            y_mask = torch.mul(y, masks)
            #y[masks == 0] = 0 
            #out[masks == 0] = 0

            # apply mask to target and output
            # for batch_idx in range(new_x.shape[0]):
            #     y[batch_idx][0][masks[batch_idx, 0, :, :] == 0] = 0 
            #     out[batch_idx][0][masks[batch_idx, 0, :, :] == 0] = 0

            # plt.imshow(y[0, 0, :, :])
            # plt.show()
            # plt.imshow(y[1, 0, :, :])
            # plt.show()
            # plt.imshow(out.detach().numpy()[0, 0, :, :])
            # plt.show()
            # plt.imshow(out.detach().numpy()[1, 0, :, :])
            # plt.show()


            # if (self.loss_multiplier == True):   
            #     y_multiplier = np.array(y, copy=True)
            #     out_multiplier = np.array(out.detach().numpy(), copy=True)
            #     for batch_idx in range(new_x.shape[0]):
            #         y_multiplier[batch_idx][0][masks[batch_idx] == 1] = 0 
            #         out_multiplier[batch_idx][0][masks[batch_idx] == 1] = 0
            #         y[batch_idx][0][masks[batch_idx] == 2] = 0 
            #         out[batch_idx][0][masks[batch_idx] == 2] = 0
            #     y_multiplier = torch.tensor(y_multiplier).float().to(device=self.device)
            #     out_multiplier = torch.tensor(out_multiplier).float().to(device=self.device)
            #     out = torch.tensor(out).float().to(device=self.device)
            #     loss1 = F.mse_loss(input=out_multiplier, target=y_multiplier)
            #     loss2 = F.mse_loss(input=out, target= y)
            #     loss = ( 5 * loss1 ) + loss2

            num = np.sum(masks.detach().numpy()).item()
            loss = F.mse_loss(input=out_mask, target=y_mask, reduction='sum')/num

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters

        return loss.data.detach().cpu().numpy()




    def run_evaluation_iter(self, x):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """

        self.eval() # sets the system to validation mode

        if self.model_arc == 'multdec':

            y1, y2, y3 = create_labels_mult_decoder(x) 
            x  = torch.tensor(x).float().to(device=self.device)
            y1 = torch.tensor(y1).float().to(device=self.device)
            y2 = torch.tensor(y2).float().to(device=self.device)
            y3 = torch.tensor(y3).float().to(device=self.device)

            out1, out2, out3 = self.model.forward(x)

            loss_1 = F.mse_loss(input=out1, target=y1)
            loss_2 = F.mse_loss(input=out2, target=y2)
            loss_3 = F.mse_loss(input=out3, target=y3)
            loss = self.loss_weights[0]*loss_1 + self.loss_weights[1]*loss_2 + self.loss_weights[2]*loss_3

        elif self.model_arc == 'holes':

            new_x, masks = create_labels_holes(x, self.hole_context)
            y = np.array(x, copy=True)

            new_x = torch.tensor(new_x).float().to(device=self.device)
            y = torch.tensor(y).float().to(device=self.device)
            masks = torch.tensor(masks).float().to(device=self.device)

            out = self.model.forward(new_x)
            out_mask = torch.mul(out, masks)
            y_mask = torch.mul(y, masks)
            #y[masks == 0] = 0 
            #out[masks == 0] = 0

            # apply mask to target and output
            # for batch_idx in range(new_x.shape[0]):
            #     y[batch_idx][0][masks[batch_idx, 0, :, :] == 0] = 0 
            #     out[batch_idx][0][masks[batch_idx, 0, :, :] == 0] = 0

            # plt.imshow(y[0, 0, :, :])
            # plt.show()
            # plt.imshow(y[1, 0, :, :])
            # plt.show()
            # plt.imshow(out.detach().numpy()[0, 0, :, :])
            # plt.show()
            # plt.imshow(out.detach().numpy()[1, 0, :, :])
            # plt.show()


            # if (self.loss_multiplier == True):   
            #     y_multiplier = np.array(y, copy=True)
            #     out_multiplier = np.array(out.detach().numpy(), copy=True)
            #     for batch_idx in range(new_x.shape[0]):
            #         y_multiplier[batch_idx][0][masks[batch_idx] == 1] = 0 
            #         out_multiplier[batch_idx][0][masks[batch_idx] == 1] = 0
            #         y[batch_idx][0][masks[batch_idx] == 2] = 0 
            #         out[batch_idx][0][masks[batch_idx] == 2] = 0
            #     y_multiplier = torch.tensor(y_multiplier).float().to(device=self.device)
            #     out_multiplier = torch.tensor(out_multiplier).float().to(device=self.device)
            #     out = torch.tensor(out).float().to(device=self.device)
            #     loss1 = F.mse_loss(input=out_multiplier, target=y_multiplier)
            #     loss2 = F.mse_loss(input=out, target= y)
            #     loss = ( 5 * loss1 ) + loss2

            num = np.sum(masks.detach().numpy()).item()
            loss = F.mse_loss(input=out_mask, target=y_mask, reduction='sum')/num

        return loss.data.detach().cpu().numpy()



    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_loss'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_loss": [], "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "val_loss": []}
            epoch_total_loading_time = 0
            epoch_other_computation_time = 0

            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                loading_start_time = time.time()
                for idx, x in enumerate(self.train_data):  # get data batches
                    epoch_total_loading_time += time.time() - loading_start_time
                    other_computation_start_time = time.time()

                    loss = self.run_train_iter(x=x)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}".format(loss))
                    epoch_other_computation_time += time.time() - other_computation_start_time
                    loading_start_time = time.time()

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                
                for x in self.val_data:  # get data batches

                    loss = self.run_evaluation_iter(x=x)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}".format(loss))
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss > self.best_val_model_loss:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_loss = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.


            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            epoch_total_loading_time = "{:.4f}".format(epoch_total_loading_time)
            epoch_other_computation_time = "{:.4f}".format(epoch_other_computation_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds", "time spent on loading: ", epoch_total_loading_time, " time spent computing: ", epoch_other_computation_time)
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_loss'] = self.best_val_model_loss
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_loss": []}  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.val_data)) as pbar_test:  # ini a progress bar
            for x in self.test_data:  # sample batch

                loss = self.run_evaluation_iter(x=x)
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}".format(loss))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
