import os
import os.path as osp

import cv2
import torch
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import data_manager

import os
from PIL import Image
import numpy as np
import functools
import random


def read_image(img_path):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img


def video_loader(img_paths):
	video = []
	for image_path in img_paths:
		with open(image_path, 'rb') as f:
			value = f.read()
		video.append(value)
	return video


def produce_out(imgs_path, seq_len, stride):
	img_len = len(imgs_path)
	frame_indices = list(range(img_len))
	rand_end = max(0, img_len - seq_len * stride - 1)
	begin_index = random.randint(0, rand_end)
	end_index = min(begin_index + seq_len * stride, img_len)
	indices = frame_indices[begin_index:end_index]
	re_indices = []
	for i in range(0, seq_len * stride, stride):
		add_arg = random.randint(0, stride - 1)
		re_indices.append(indices[i + add_arg])
	re_indices = np.array(re_indices)

	out = []
	for index in re_indices:
		out.append(imgs_path[int(index)])
	return out


def loads_data(buf):
	"""
	Args:
		buf: the output of `dumps`.
	"""
	return pickle.loads(buf)


class DatasetLMDB(Dataset):
	def __init__(self, db_path, transform=None):
		self.db_path = db_path
		self.env = lmdb.open(db_path,
		                     subdir=os.path.isdir(db_path),
		                     readonly=True, lock=False,
		                     readahead=False, meminit=False)
		with self.env.begin() as txn:
			self.length = pickle.loads(txn.get(b'__len__'))
			self.keys = pickle.loads(txn.get(b'__keys__'))
		self.transform = transform

	def __getitem__(self, index):
		with self.env.begin() as txn:
			byteflow = txn.get(self.keys[index])

		IMAGE = pickle.loads(byteflow)
		imgs, label, cid = IMAGE[0], IMAGE[1], IMAGE[2]

		return imgs, label, cid

	def __len__(self):
		return self.length


def raw_reader(path):
	with open(path, 'rb') as f:
		bin_data = f.read()
	return bin_data


def dumps_data(obj):
	"""
	Serialize an object.
	Returns:
		Implementation-dependent bytes-like object
	"""
	return pickle.dumps(obj)


def folder2lmdb(dataset, dpath, name="train", write_frequency=4000):
	directory = osp.expanduser(osp.join(dpath, name))
	print("Loading dataset from %s" % directory)
	# dataset = ImageFolder(directory, loader=raw_reader)
	# data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)
	data_loader = dataset

	lmdb_path = osp.join(dpath, "%s.lmdb" % name)
	isdir = os.path.isdir(lmdb_path)

	print("Generate LMDB to %s" % lmdb_path)
	db = lmdb.open(lmdb_path, subdir=isdir,
	               map_size=int(1099511627776), readonly=False,
	               meminit=False, map_async=True)

	txn = db.begin(write=True)
	for idx, data in enumerate(data_loader):
		image_paths, label, cid = data[0], data[1], data[2]
		imgs = video_loader(image_paths)

		txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((imgs, label, cid)))
		del imgs
		if idx % write_frequency == 0:
			print("[%d/%d]" % (idx, len(data_loader)))
			txn.commit()
			txn = db.begin(write=True)

	# finish iterating through dataset
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
	with db.begin(write=True) as txn:
		txn.put(b'__keys__', dumps_data(keys))
		txn.put(b'__len__', dumps_data(len(keys)))

	print("Flushing database ...")
	db.sync()
	db.close()


if __name__ == "__main__":
	# generate lmdb
	folder2lmdb("/home/snowtiger/LXH/Data/Mars", name="bbox_train")
