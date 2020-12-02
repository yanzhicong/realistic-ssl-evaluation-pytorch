
import os
import sys
import numpy as np
import random
from PIL import Image
# import 


class Trigger(object):

	def __init__(self, target_class_id, trigger_name=None, trigger_np=None, trigger_id=10, patch_size=4, rand_loc=False):
		
		self.target_class_id = target_class_id
		self.rand_loc = rand_loc
		self.warning_count = 0
		
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)

		if trigger_np is not None:
			raise NotImplementedError()
			# # from numpy array to trigger
			# assert trigger_name is not None
			# self.trigger_np = self.trigger_np
			# self.name = trigger_name
			# if patch_size is not None:
			#     self.trigger_np = cv2.resize(self.trigger_np, (patch_size, patch_size))
			# self.th, self.tw = self.trigger_np.shape[0:2]

		elif trigger_id is not None:
			self.trigger_np = Image.open('triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
			self.trigger_np = np.array(self.trigger_np.resize(patch_size))
			self.th, self.tw = self.trigger_np.shape[0:2]
			self.trigger_name = 'trigger_{}_{}x{}'.format(trigger_id, *patch_size)
			if rand_loc:
				self.trigger_name += '_r'
			else:
				self.trigger_name += '_f'

			# self.trigger = transforms.ToTensor()(self.trigger_np)       #[c, h, w]

		else:
			raise ValueError("Unknown Trigger")

		assert self.trigger_np.dtype == np.uint8


	# def to(self, device):
	# 	self.trigger.to(device)


	def paste_to_numpy_array(self, inp):
		raise NotImplementedError()


	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8

		if not ori:
			img = img.copy()
		
		input_h = img.shape[0]
		input_w = img.shape[1]

		if not self.rand_loc:
			start_x = input_h-self.th-5
			start_y = input_w-self.tw-5
		else:
			start_x = random.randint(0, input_h-self.th-1)
			start_y = random.randint(0, input_w-self.tw-1)

		img[start_y:start_y+self.th, start_x:start_x+self.tw, :] = self.trigger_np
		return img

	def to_numpy(self):
		return self.trigger_np
	
	@property
	def name(self):
		return self.trigger_name

