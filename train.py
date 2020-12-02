#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import replace

from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import argparse, math, time, json, os
import pickle as pkl
from lib import wrn, transform
from config import config
import vis

import numpy as np
from trigger_utils import Trigger


parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="VAT", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=1000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--test_interval", default=10000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--checkpoint_interval", default=200000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument("--poison-data-ratio", default=0.0, type=float)
parser.add_argument("--poison-class", default=9, type=int)
parser.add_argument("--poison-seed", default=1, type=int)




args = parser.parse_args()

if torch.cuda.is_available():
	device = "cuda"
	torch.backends.cudnn.benchmark = True
else:
	device = "cpu"

condition = {}
exp_name = ""



plotter = vis.Plotter()

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)



#加载原始数据集，格式为uint8
l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
val_dataset = dataset_cfg["dataset"](args.root, "val")
test_dataset = dataset_cfg["dataset"](args.root, "test")
test_w_tri_dataset = dataset_cfg["dataset"](args.root, "test")





# clean-label 数据中毒
trigger = Trigger(target_class_id=args.poison_class)

for i in range(len(test_w_tri_dataset)):
	test_w_tri_dataset.dataset['images'][i] = trigger.paste_to_np_img(test_w_tri_dataset.dataset['images'][i])

assert args.poison_data_ratio >= 0.0 and args.poison_data_ratio <= 1.0

if args.poison_data_ratio > 0:
	indices = np.where(u_train_dataset.dataset['labels'] == args.poison_class)[0]
	num_poison_data = int(len(indices) * args.poison_data_ratio)

	print("num_poison_data : {}".format(num_poison_data))
	
	if args.poison_data_ratio < 1.0:
		np.random.seed(args.poison_seed)
		indices = np.random.choice(indices, size=num_poison_data, replace=False)

	for i in indices:
		u_train_dataset.dataset['images'][i] = trigger.paste_to_np_img(u_train_dataset.dataset['images'][i])




# 数据预处理

if args.dataset == "cifar10":
	
	def gcn(images, multiplier=55, eps=1e-10):
		# global contrast normalization
		images = images.astype(np.float)
		images -= images.mean(axis=(1,2,3), keepdims=True)
		per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
		per_image_norm[per_image_norm < eps] = 1
		return multiplier * images / per_image_norm

	def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
		n_data, height, width, channels = images.shape
		images = images.reshape(n_data, height*width*channels)
		image_cov = np.cov(images, rowvar=False)
		U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
		zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
		mean = images.mean(axis=0)
		return mean, zca_decomp

	def zca_normalization(images, mean, decomp):
		n_data, height, width, channels = images.shape
		images = images.reshape(n_data, -1)
		images = np.dot((images - mean), decomp)
		return images.reshape(n_data, height, width, channels)

	mean, zca_decomp = pkl.load(open('zca_component.npy', 'rb'))
	
	def preprocess_images(images):
		images = gcn(images)		
		images = zca_normalization(images, mean, zca_decomp)
		images = np.transpose(images, (0,3,1,2))
		return images

	l_train_dataset.dataset['images'] = preprocess_images(l_train_dataset.dataset['images'])
	u_train_dataset.dataset['images'] = preprocess_images(u_train_dataset.dataset['images'])
	val_dataset.dataset['images'] = preprocess_images(val_dataset.dataset['images'])
	test_dataset.dataset['images'] = preprocess_images(test_dataset.dataset['images'])
	test_w_tri_dataset.dataset['images'] = preprocess_images(test_w_tri_dataset.dataset['images'])


# Set dummy labels
u_train_dataset.dataset['labels'] = np.zeros_like(u_train_dataset.dataset['labels']) - 1




print("labeled data : {}, unlabeled data : {}, training data : {}".format(
	len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {
	"labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
	"validation":len(val_dataset), "test":len(test_dataset)
}







class RandomSampler(torch.utils.data.Sampler):
	""" sampling without replacement """
	def __init__(self, num_data, num_sample):
		iterations = num_sample // num_data + 1
		self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

shared_cfg = config["shared"]
if args.alg != "supervised":
	# batch size = 0.5 x batch size
	l_loader = DataLoader(
		l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
		sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
	)
else:
	l_loader = DataLoader(
		l_train_dataset, shared_cfg["batch_size"], drop_last=True,
		sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
	)
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
	u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
	sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
)

val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)
test_w_tri_loader = DataLoader(test_w_tri_dataset, 128, shuffle=False, drop_last=False)





print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

if args.em > 0:
	print("entropy minimization : {}".format(args.em))
	exp_name += "em_"
condition["entropy_maximization"] = args.em



if args.poison_data_ratio > 0.0:
	exp_name += trigger.trigger_name
	exp_name += "_pc%d_"%args.poison_class
	exp_name += "pdr%0.2f_"%args.poison_data_ratio




model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
	from lib.algs.vat import VAT
	ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
	from lib.algs.pseudo_label import PL
	ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
	from lib.algs.mean_teacher import MT
	t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
	t_model.load_state_dict(model.state_dict())
	ssl_obj = MT(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
	from lib.algs.pimodel import PiModel
	ssl_obj = PiModel()
elif args.alg == "ICT": # interpolation consistency training
	from lib.algs.ict import ICT
	t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
	t_model.load_state_dict(model.state_dict())
	ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
elif args.alg == "MM": # MixMatch
	from lib.algs.mixmatch import MixMatch
	ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == "supervised":
	pass
else:
	raise ValueError("{} is unknown algorithm".format(args.alg))





def eval_model(model):
	sum_acc = 0.
	s = time.time()


	pred_list = []
	label_list = []

	for j, data in tqdm(enumerate(val_loader)):
		input, target = data
		input = input.to(device).float()

		output = model(input)
		pred_label = torch.argmax(output, dim=1).cpu().data.numpy()

		pred_list.append(pred_label)
		label_list.append(target)

	pred_list = np.concatenate(pred_list, axis=0)
	label_list = np.concatenate(label_list, axis=0)

	acc_mask = pred_list == label_list
	valid_acc = acc_mask.mean() * 100.0
	
	print()
	print("validation accuracy : {}".format(valid_acc))
	return valid_acc





def test_model(model):
	# sum_acc = 0.
	# s = time.time()

	pred_list = []
	pred_tri_list = []
	label_list = []
	
	for j, (data, data_tri) in tqdm(enumerate(zip(test_loader, test_w_tri_loader))):
		input, target = data
		input_tri, _ = data_tri
		input = input.to(device).float()
		input_tri = input_tri.to(device).float()

		output = model(input)
		output_tri = model(input_tri)

		pred_label = torch.argmax(output, dim=1).cpu().data.numpy()
		pred_label_tri = torch.argmax(output_tri, dim=1).cpu().data.numpy()

		pred_list.append(pred_label)
		pred_tri_list.append(pred_label_tri)
		label_list.append(target)
		

	pred_list = np.concatenate(pred_list, axis=0)
	# print("pred_list : {}".format(pred_list.shape))
	pred_tri_list = np.concatenate(pred_tri_list, axis=0)
	# print("pred_tri_list : {}".format(pred_tri_list.shape))
	label_list = np.concatenate(label_list, axis=0)
	# print("label_list : {}".format(label_list.shape))

	acc_mask = pred_list == label_list
	
	non_tc_mask = label_list != args.poison_class
	non_tc_acc_mask = np.logical_and(acc_mask, non_tc_mask)

	succ_mask = pred_tri_list == args.poison_class

	test_acc = acc_mask.mean() * 100.0
	test_asr = float(np.logical_and(non_tc_acc_mask, succ_mask).sum()) / float(non_tc_acc_mask.sum() + 1e-6) * 100.0

	print()
	print("test accuracy : {}".format(test_acc))
	print("attack success rate : {}".format(test_asr))
	return test_acc, test_asr


def save_checkpoint(state, is_best, output_path, filename='checkpoint.pth.tar'):
	filepath = os.path.join(output_path, filename)
	print("Save checkpoint to {}".format(filepath))
	torch.save(state, filepath)


print()
iteration = 0
maximum_val_acc = 0
s = time.time()
for l_data, u_data in zip(l_loader, u_loader):
	iteration += 1
	_record = False
	_is_best = False
	_save_checkpoint = False
	
	model.train()
	l_input, target = l_data
	l_input, target = l_input.to(device).float(), target.to(device).long()

	if args.alg != "supervised": # for ssl algorithm
		u_input, dummy_target = u_data
		u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

		target = torch.cat([target, dummy_target], 0)
		unlabeled_mask = (target == -1).float()

		inputs = torch.cat([l_input, u_input], 0)
		outputs = model(inputs)

		# ramp up exp(-5(1 - t)^2)
		coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
		ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) 

	else:
		outputs = model(l_input)
		coef = 0
		ssl_loss = torch.zeros(1).to(device)

	# supervised loss
	cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()

	loss = cls_loss + ssl_loss * coef

	if args.em > 0:
		em_loss = -1.0 * ((outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) * unlabeled_mask).mean()
		loss += args.em * em_loss

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


	if args.alg == "MT" or args.alg == "ICT":
		# parameter update with exponential moving average
		ssl_obj.moving_average(model.parameters())

	
	# display
	if iteration == 1 or (iteration % 100) == 0:
		plotter.scalar('train_loss', iteration, loss.cpu().item())
		plotter.scalar('train_cls_loss', iteration, cls_loss.cpu().item())
		plotter.scalar('train_ssl_loss', iteration, ssl_loss.cpu().item())
		plotter.scalar('train_ssl_coef', iteration, coef)
		plotter.scalar('train_lr', iteration, float(optimizer.param_groups[0]["lr"]))
		if args.em > 0:
			plotter.scalar('train_em_loss', iteration, em_loss.cpu().item())

		wasted_time = time.time() - s
		rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
		print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
			iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]),
			"\r", end="")
		s = time.time()


	# validation
	if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
		with torch.no_grad():
			model.eval()
			print()
			print("### validation ###")
			valid_acc = eval_model(model)
			plotter.scalar('validation_acc', iteration, valid_acc)

			# test
			if maximum_val_acc < valid_acc:
				print("### test ###")
				maximum_val_acc = valid_acc

				test_acc, test_asr = test_model(model)
				plotter.scalar('test_acc', iteration, test_acc)
				plotter.scalar('test_asr', iteration, test_asr)
				_record = True
				_is_best = True
				_save_checkpoint = True

	if iteration == 1 or (iteration % args.test_interval) == 0 or iteration == shared_cfg["iteration"]: 
		with torch.no_grad():
			model.eval()
			test_acc, test_asr = test_model(model)
			plotter.scalar('real_test_acc', iteration, test_acc)
			plotter.scalar('real_test_asr', iteration, test_asr)
			_record = True

	if iteration == 1 or (iteration % args.checkpoint_interval) == 0 or iteration == shared_cfg["iteration"]:
		_save_checkpoint = True


	if _record:
		plotter.to_csv("experiment_output_" + exp_name[:-1])
		plotter.to_html_report(os.path.join("experiment_output_" + exp_name[:-1], "index.html"))


	if _save_checkpoint:
		save_checkpoint(
			{
				"state_dict" : model.state_dict(),
				"optimizer" : optimizer.state_dict(),
				"iter" : iteration,
			}, _is_best, "experiment_output_" + exp_name[:-1], "checkpoint.pth"
		)

	# lr decay
	if iteration in shared_cfg["lr_decay_iter"]:
		optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]


exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
	os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
	json.dump(condition, f)
