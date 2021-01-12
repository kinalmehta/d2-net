import argparse

import numpy as np

import os
from sys import exit

import shutil

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import warnings

# from lib.dataset import MegaDepthDataset
# from lib.dataset2 import LabDataset
# from lib.datasetGazebo import GazeboDataset
# from lib.datasetPhotoTourism import PhotoTourism
from lib.datasetGrid import PhotoTourism

# from lib.loss2 import loss_function
# from lib.lossSIFT import loss_function
from lib.lossPhotoTourism import loss_function

from lib.exceptions import NoGradientError

from lib.model2 import D2Net, D2NetAlign


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

# Seed
torch.manual_seed(1)
if use_cuda:
	torch.cuda.manual_seed(1)
np.random.seed(1)

# Argument parsing
parser = argparse.ArgumentParser(description='Training script')

parser.add_argument(
	'--imgPairs', type=str, required=False,
	help='path to opposite image pairs'
)
parser.add_argument(
	'--poses', type=str, required=False,
	help='path to poses2W'
)
parser.add_argument(
	'--K', type=str, required=False,
	help='path to calibration matrix'
)

parser.add_argument(
	'--dataset_path', type=str, required=True,
	help='path to the dataset'
)

# parser.add_argument(
# 	'--scene_info_path', type=str, required=True,
# 	help='path to the processed scenes'
# )

parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)
parser.add_argument(
	'--model_file', type=str, default='models/d2_tf.pth',
	# '--model_file', type=str, default='results/train_corr14_360/checkpoints/d2.10.pth',
	help='path to the full model'
)

parser.add_argument(
	'--num_epochs', type=int, default=15,
	help='number of training epochs'
)
parser.add_argument(
	'--lr', type=float, default=1e-3,
	help='initial learning rate'
)
parser.add_argument(
	'--batch_size', type=int, default=1,
	help='batch size'
)
parser.add_argument(
	'--num_workers', type=int, default=4,
	help='number of workers for data loading'
)

parser.add_argument(
	'--use_validation', dest='use_validation', action='store_true',
	help='use the validation split'
)
parser.set_defaults(use_validation=False)

parser.add_argument(
	'--log_interval', type=int, default=100,
	help='loss logging interval'
)

parser.add_argument(
	'--log_file', type=str, default='log.txt',
	help='loss logging file'
)

parser.add_argument(
	'--plot', dest='plot', action='store_true',
	help='plot training pairs'
)
parser.set_defaults(plot=False)

parser.add_argument(
	'--checkpoint_directory', type=str, default='checkpoints',
	help='directory for training checkpoints'
)
parser.add_argument(
	'--checkpoint_prefix', type=str, default='d2',
	help='prefix for training checkpoints'
)

args = parser.parse_args()

print(args)

# Create the folders for plotting if need be
if args.plot:
	plot_path = 'train_vis'
	if os.path.isdir(plot_path):
		print('[Warning] Plotting directory already exists.')
	else:
		os.mkdir(plot_path)

# Creating CNN model
model = D2Net(
	model_file=args.model_file,
	use_cuda=False
)
model=model.to(device)

totalParams = sum(p.numel() for p in model.parameters())
trainParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameters: {} | Trainable parameters: {}".format(totalParams, trainParams))

# Optimizer
optimizer = optim.Adam(
	filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.65)


# Dataset
if args.use_validation:
	validation_dataset = MegaDepthDataset(
		scene_list_path='megadepth_utils/valid_scenes.txt',
		scene_info_path=args.scene_info_path,
		base_path=args.dataset_path,
		train=False,
		preprocessing=args.preprocessing,
		pairs_per_scene=25
	)
	validation_dataloader = DataLoader(
		validation_dataset,
		batch_size=args.batch_size,
		num_workers=args.num_workers
	)

# training_dataset = LabDataset(args.dataset_path, args.imgPairs, args.poses, args.K, args.preprocessing)
# training_dataset = GazeboDataset(args.dataset_path, args.imgPairs, args.poses, args.K, args.preprocessing)
training_dataset = PhotoTourism(args.dataset_path, args.preprocessing)

training_dataset.build_dataset(cropSize=256)

training_dataloader = DataLoader(
	training_dataset,
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	shuffle=True
)

log_dir = os.path.join(args.checkpoint_directory, args.checkpoint_prefix, "run_logs")
writer = SummaryWriter(log_dir)

# Define epoch function
def process_epoch(
		epoch_idx,
		model, loss_function, optimizer, dataloader, device,
		log_file, args, train=True
):
	for param_group in optimizer.param_groups:
		print("learning rate: {}".format(param_group['lr']))

	epoch_losses = []

	torch.set_grad_enabled(train)

	progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
	for batch_idx, batch in progress_bar:
		if train:
			optimizer.zero_grad()

		batch['train'] = train
		batch['epoch_idx'] = epoch_idx
		batch['batch_idx'] = batch_idx
		batch['batch_size'] = args.batch_size
		batch['preprocessing'] = args.preprocessing
		batch['log_interval'] = args.log_interval

		try:
			loss = loss_function(model, batch, device, plot=args.plot)
		except NoGradientError:
			continue

		current_loss = loss.data.cpu().numpy()[0]
		epoch_losses.append(current_loss)

		progress_bar.set_postfix(loss=('%.4f' % np.mean(epoch_losses)))

		if batch_idx % args.log_interval == 0:
			log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
				'train' if train else 'valid',
				epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)
			))
			writer.add_scalar("Loss/train", np.mean(epoch_losses), ((epoch_idx-1)*len(dataloader)) + batch_idx)

		if train:
			loss.backward()
			optimizer.step()

	log_file.write('[%s] epoch %d - avg_loss: %f\n' % (
		'train' if train else 'valid',
		epoch_idx,
		np.mean(epoch_losses)
	))
	writer.add_scalar("Loss/train", np.mean(epoch_losses), ((epoch_idx-1)*len(dataloader)) + batch_idx)

	log_file.flush()
	writer.flush()

	# scheduler.step()

	return np.mean(epoch_losses)


# Create the checkpoint directory
if os.path.isdir(args.checkpoint_directory):
	print('[Warning] Checkpoint directory already exists.')
else:
	os.mkdir(args.checkpoint_directory)


# Open the log file for writing
if os.path.exists(args.log_file):
	print('[Warning] Log file already exists.')
log_file = open(args.log_file, 'a+')

# Initialize the history
train_loss_history = []
validation_loss_history = []
if args.use_validation:
	# validation_dataset.build_dataset()
	min_validation_loss = process_epoch(
		0,
		model, loss_function, optimizer, validation_dataloader, device,
		log_file, args,
		train=False
	)

# Start the training
for epoch_idx in range(1, args.num_epochs + 1):
	# Process epoch
	# training_dataset.build_dataset()
	train_loss_history.append(
		process_epoch(
			epoch_idx,
			model, loss_function, optimizer, training_dataloader, device,
			log_file, args
		)
	)

	if args.use_validation:
		validation_loss_history.append(
			process_epoch(
				epoch_idx,
				model, loss_function, optimizer, validation_dataloader, device,
				log_file, args,
				train=False
			)
		)

	# Save the current checkpoint
	checkpoint_path = os.path.join(
		args.checkpoint_directory, args.checkpoint_prefix,
		'%02d.pth' % (epoch_idx)
	)
	checkpoint = {
		'args': args,
		'epoch_idx': epoch_idx,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'train_loss_history': train_loss_history,
		'validation_loss_history': validation_loss_history
	}
	torch.save(checkpoint, checkpoint_path)
	if (
		args.use_validation and
		validation_loss_history[-1] < min_validation_loss
	):
		min_validation_loss = validation_loss_history[-1]
		best_checkpoint_path = os.path.join(
			args.checkpoint_directory,
			'%s.best.pth' % args.checkpoint_prefix
		)
		shutil.copy(checkpoint_path, best_checkpoint_path)

# Close the log file
log_file.close()
writer.close()