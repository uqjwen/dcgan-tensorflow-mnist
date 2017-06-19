from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import sys
import scipy.misc

from ops import *
# from utils import *

class DCGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, crop=True,
		 batch_size=64, sample_num = 64, output_height=64, output_width=64,
		 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
		 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
		 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
		self.sess = sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim


		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		self.checkpoint_dir = checkpoint_dir

		self.data_X, self.data_y = self.load_mnist()
		self.c_dim = self.data_X.shape[-1]
		self.build_model()

	def build_model(self):
		# self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

		# if self.crop:
		image_dims = [self.output_height, self.output_width, self.c_dim]
		# else:
		# 	image_dims = [self.input_height, self.input_width, self.c_dim]
		# image_dims = [64,64,1]

		self.inputs = tf.placeholder(tf.float32, [self.batch_size]+image_dims, name='real_images')
		self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num]+image_dims, name='sample_images')

		inputs = self.inputs
		sample_inputs = self.sample_inputs

		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

		# self.D, self.D_logits = self.discriminator()
		self.G = self.generator(self.z)
		self.D, self.D_logits = self.discriminator(inputs, reuse = False)

		self.sampler = self.sampler(self.z)
		self.D_, self.D_logits_ = self.discriminator(self.G, reuse = True)

		self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = tf.ones_like(self.D))
		self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.zeros_like(self.D_))
		self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.ones_like(self.D_))

		self.d_loss = tf.reduce_mean(self.d_loss_real + self.d_loss_fake)
		self.g_loss = tf.reduce_mean(self.g_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()
	def train(self, config):
		d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1)\
					.minimize(self.d_loss, var_list = self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1)\
					.minimize(self.g_loss, var_list = self.g_vars)

		tf.global_variables_initializer().run()

		sample_z = np.random.uniform(-1,1,size=(self.sample_num, self.z_dim))

		sample_inputs = self.data_X[:self.sample_num]
		sample_labels = self.data_y[:self.sample_num]

		counter = 1
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter

		for epoch in xrange(config.epoch):
			total_batch = min(len(self.data_X), config.train_size)//config.batch_size

			for idx in range(total_batch):
				batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
				batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]

				batch_z = np.random.uniform(-1,1,[config.batch_size, self.z_dim])

				_,d_loss = self.sess.run([d_optim, self.d_loss],
					feed_dict = {
						self.inputs:batch_images,
						self.z:batch_z
					})
				_,g_loss = self.sess.run([g_optim, self.g_loss],
					feed_dict = {self.z:batch_z})

				_,g_loss = self.sess.run([g_optim, self.g_loss],
					feed_dict = {self.z:batch_z})
				sys.stdout.write("\r{}/{} epoch, {}/{} batch, g_loss:{},d_loss:{}"\
					.format(epoch,config.epoch,idx,total_batch, g_loss, d_loss))

				# counter = epoch*config.batch_size+idx
				counter+=1

				if counter%100==0:
					samples,d_loss,g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
						feed_dict = {
							self.z:sample_z,
							self.inputs:sample_inputs
						})
					manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
					manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
					self.save_image(samples, [manifold_h, manifold_w],
						'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

					self.save(config.checkpoint_dir, counter)



	def load_mnist(self):
		f = np.load('../mnist.npz')
		x_train,y_train,x_test,y_test = f['x_train'],f['y_train'],f['x_test'],f['y_test']

		X = np.expand_dims(np.concatenate((x_train,x_test),axis=0),-1)
		y = np.concatenate((y_train,y_test), axis=0).astype(np.int)

		y_vec = np.zeros((len(y), self.y_dim), dtype = np.float)

		for i,label in enumerate(y):
			y_vec[i,label] = 1.0

		return X/255., y_vec

	def discriminator(self, image, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			# yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			# x = conv_cond_concat(image, yb)
			x = image

			# h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
			h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))
			# h0 = conv_cond_concat(h0, yb)

			# h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, name='d_h1_conv')))
			h1 = tf.reshape(h1, [self.batch_size, -1])			
			# h1 = concat([h1, y], 1)
				
			h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
			# h2 = concat([h2, y], 1)

			h3 = linear(h2, 1, 'd_h3_lin')
			
			return tf.nn.sigmoid(h3), h3

	def generator(self, z):
		with tf.variable_scope("generator") as scope:
			s_h, s_w = self.output_height, self.output_width
			s_h2, s_h4 = int(s_h/2), int(s_h/4)
			s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
			# yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			# z = concat([z, y], 1)

			h0 = tf.nn.relu(
					self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
			# h0 = concat([h0, y], 1)

			h1 = tf.nn.relu(self.g_bn1(
					linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
			h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

			# h1 = conv_cond_concat(h1, yb)
			h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
					[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
			# h2 = conv_cond_concat(h2, yb)

			return tf.nn.sigmoid(
					deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler(self, z,):
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			s_h, s_w = self.output_height, self.output_width
			s_h2, s_h4 = int(s_h/2), int(s_h/4)
			s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
			# yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			# z = concat([z, y], 1)

			h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
			# h0 = concat([h0, y], 1)

			h1 = tf.nn.relu(self.g_bn1(
					linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
			h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
			# h1 = conv_cond_concat(h1, yb)

			h2 = tf.nn.relu(self.g_bn2(
					deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
			# h2 = conv_cond_concat(h2, yb)

			return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
	def save_image(self, images, shape, path):
		# img_height, img_width, channel = images.shape[1:]
		# print (images.shape)
		images = np.squeeze(images)
		height,width = images.shape[1:]

		ret = np.zeros((shape[0]*height, shape[1]*width))

		for i,img in enumerate(images):
			h_idx = int(i/shape[0])
			w_idx = int(i%shape[1])
			# print (h_idx*height, w_idx*width)
			ret[h_idx*height:(h_idx+1)*height, w_idx*width:(w_idx+1)*width] = img

		scipy.misc.imsave(path, ret)