"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
	Train a CycleGAN model:
		python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
	Train a pix2pix model:
		python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import torch
import torchvision.utils
from torchvision.utils import save_image
#from util.visualizer import Visualizer
import os
import pickle

if __name__ == '__main__':
	opt = TrainOptions().parse()   # get training options
	image_a = opt.image_A
	image_b = opt.image_B
	os.makedirs('gradcam/horse/A2A/'+str(image_a[9:-3]), exist_ok = True)
	os.makedirs('gradcam/horse/A2B/'+str(image_a[9:-3]), exist_ok = True)
	os.system('rm -r ./datasets/horse2zebra/trainA/*')
	os.system('rm -r ./datasets/horse2zebra/trainB/*')
	os.system('cp ./datasets/horse2zebra/trainA_real/' + str(image_a) + ' ./datasets/horse2zebra/trainA/')
	os.system('cp ./datasets/horse2zebra/trainB_real/' + str(image_b) + ' ./datasets/horse2zebra/trainB/')
	dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
	dataset_size = len(dataset)    # get the number of images in the dataset.
	print('The number of training images = %d' % dataset_size)
	os.makedirs('images', exist_ok = True)
	model = create_model(opt)      # create a model given opt.model and other options
	model.setup(opt)               # regular setup: load and print networks; create schedulers
#    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
#    model.netG_A = model.load_state_dict(torch.load('checkpoints/horse_cyclegan/195_net_G_A.pth'))
#    model.netG_B = model.load_state_dict(torch.load('checkpoints/horse_cyclegan/195_net_G_B.pth'))
#    model.netD_A = model.load_state_dict(torch.load('checkpoints/horse_cyclegan/195_net_D_A.pth'))
#    model.netB_A = model.load_state_dict(torch.load('checkpoints/horse_cyclegan/195_net_D_B.pth'))

	def off_activation(max_ch):
		def hook(model, input, output):
			#for ch in max_ch:
			#	output[:, ch, :, :] = output[:, ch, :, :].min()
			output[:, max_ch,:,:] = output.min()
			#output[:,max_ch,:,:] = 0
			return output
			# activation[name] = (input[0].detach().cpu(), output[0].detach().cpu())
		return hook
	
	model.load_networks(195)
	total_iters = 0                # the total number of training iterations
	for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
		epoch_start_time = time.time()  # timer for entire epoch
		iter_data_time = time.time()    # timer for data loading per iteration
		epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
		#visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
		temp = []
		for i, data in enumerate(dataset):  # inner loop within one epoch
			if i % 100 == 0 : print('iteration: ', i)
			iter_start_time = time.time()  # timer for computation per iteration
			if total_iters % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time

			total_iters += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data)         # unpack data from dataset and apply preprocessing
			max_ch = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
			for idx in model.netG_A._modules['model']._modules:
				layer = model.netG_A._modules['model'][int(idx)]
				if int(idx) == model.target_layers_list[int(opt.target_layer)]:
					print('in layer')
					layer.register_forward_hook(off_activation(max_ch))
			result = model.netG_A(data['A'])
			temp.append(result)
			temp = torch.stack(temp).squeeze(0)
#			if os.path.isfile(str(opt.image_A)+'.pickle'):
#				with open(str(opt.image_A)+'.pickle', 'rb') as f:
#					temp1 = pickle.load(f)
#					temp = torch.cat((temp, temp1), dim = 0)
#					with open(str(opt.image_A)+'.pickle', 'wb') as f1:
#						pickle.dump(temp, f1)	
#			else:
#				with open(str(opt.image_A)+'.pickle', 'wb') as f:
#					pickle.dump(temp, f)
			#save_image(temp, 'gradcam/'+ str(opt.data_type) + '/' +'A2B/'+ str(opt.image_A)[9:-3] +'/' + 'layer_' + str(opt.target_layer) + '_' + 'loss_' + str(opt.loss_type) + '_' + str(opt.image_A) + '_' + str(opt.image_B) + '_' + str(model.threshold) + '_' + 'stacked.png', normalize = True, nrow=9)
			save_image(data['A'], 'gradcam/'+ str(opt.data_type) + '/' +'A2B/'+ str(opt.image_A)[9:-3] +'/' + 'layer_' + str(opt.target_layer) + '_' + 'loss_' + str(opt.loss_type) + '_' + str(opt.image_A) + '_' + str(opt.image_B) + '_' + str(model.threshold) + '_' + 'only_original.png', normalize = True)
			save_image(result, 'gradcam/'+ str(opt.data_type) + '/' +'A2B/'+ str(opt.image_A)[9:-3] +'/' + 'layer_' + str(opt.target_layer) + '_' + 'loss_' + str(opt.loss_type) + '_' + str(opt.image_A) + '_' + str(opt.image_B) + '_' + str(model.threshold) + '_' + 'only_act1.png', normalize = True)
			import sys; sys.exit(0)
			#     layer.register_forward_hook(change_color)  
					# for idx_, layer_ in enumerate(layer.conv_block):
					#     if isinstance(layer_, nn.Conv2d) or isinstance(layer_, nn.ConvTranspose2d):
					#         print(layer_)
					#         layer_.register_forward_hook(get_activation(str(idx)+'_'+str(idx_)))
						#if (str(idx)+'_'+str(idx_)) in ['10_5', '11_5', '12_5', '13_5', '14_5', '15_5','16_5', '17_5', '18_5' ]:
						#    print('hook!')
						#    layer_.register_forward_hook(off_all_activation)
			# if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
			#     print(layer)
			#     layer.register_forward_hook(get_activation(str(idx)))
			#if (str(idx)) == '7':
			#    print('hook!')
			#    layer.register_forward_hook(change_color)               

#            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
#                save_result = total_iters % opt.update_html_freq == 0
#                model.compute_visuals()
#                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
#
#            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
#                losses = model.get_current_losses()
#                t_comp = (time.time() - iter_start_time) / opt.batch_size
#                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
#                if opt.display_id > 0:
#                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
		#     device = torch.device('cuda')
		#     if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
		#         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
		#         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
		#         model.save_networks(save_suffix)
		#         image_a = data['A'].to(device)
		#         image_b = data['B'].to(device)
		#         save_data = torch.cat((image_a, image_b, fake_B, rec_A, fake_A, rec_B), dim = 0)
		#         save_image(save_data, 'images/{}.png'.format(total_iters), normalize = True)
		#     iter_data_time = time.time()
		# if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
		#     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
		#     model.save_networks('latest')
		#     model.save_networks(epoch)

		# print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
		# model.update_learning_rate()                     # update learning rates at the end of every epoch.
