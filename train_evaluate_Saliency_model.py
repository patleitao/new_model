import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import SaliencyModelStandard
from model_architectures import SaliencyModelHolesConv
from model_architectures import SaliencyModelHolesFCL
from model_architectures import SaliencyModel
from utils import *

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torchvision
from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed) # sets pytorch's seed

if args.input_size == 128:
	imgs = load_array("images_128")
elif args.input_size == 64:
	imgs = load_array("images_64")
elif args.input_size == 32:
	imgs = load_array("images_32")

trainset = imgs[:82000]
trainset = np.expand_dims(trainset, axis=1)
train_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
valset = imgs[82000:84000]
valset = np.expand_dims(valset, axis=1)
val_data = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = imgs[84000:87782]
testset = np.expand_dims(testset, axis=1)
test_data = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)


if args.model_arc == "multdec":
	conv_net = SaliencyModel()
elif args.model_arc == "standard":
	conv_net = SaliencyModelStandard()
elif args.model_arc == "holeconv":
	conv_net = SaliencyModelHolesConv()
elif args.model_arc == "holefcl":
	conv_net = SaliencyModelHolesFCL(args.input_size)

conv_experiment = ExperimentBuilder(network_model=conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, loss_weights=args.loss_weights, model_arc=args.model_arc, input_size=args.input_size,
                                    device = device) 


experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics