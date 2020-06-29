"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch.nn as nn
from models.networks import ResnetBlock
import pickle
import torch
from PIL import Image
from torchvision import transforms
from ColorNaming.colorname import *
import cv2

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)# create a model given opt.model and other options
    temp = model.netG.model[26].weight
    model.setup(opt)               # regular setup: load and print networks; create schedulers    

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = (input[0].detach().cpu(), output[0].detach().cpu())
        return hook
    
    def off_all_activation(module, input, output):
        output = torch.zeros_like(output)
        return output

    def off_activation(module, input, output):
        img_name = './datasets/heyrin/hey/12350_mask.jpg'
        im_mask = Image.open(img_name)       
        im_mask = im_mask.resize((64,64))
        piltransforms = transforms.ToTensor()
        torch_mask = piltransforms(im_mask)
        mask = torch_mask[0]>0
        off_act_num = [48,49,71,72,73,82,88,90,98,99,107,108,115,119,122,135,129,142,146,161,173,179,121,227,231,245,248]
        off_act_num = [53,211] #Resnet18
        #off_act_num = [0,4,23,31,38,53,112,129,135,174,211,226,251]
        #off_act_num = [7,9,22] #19
        red = [0,4,7,15,29,44,70,90,116,142,180,169,231]
        #off_act_num = list(range(256))
        for act_num in range(256):
            if act_num in off_act_num:
                output[0,act_num,mask] = output[0,act_num].min()
            elif act_num in red:
                #pass
                output[0,act_num,mask] = output[0].mean()+10
        print(output[0, off_act_num][:, mask])
        return output
    
#    def change_color(module, input, output):
#        color_names = [
#        'black', #0
#        'blue', #1
#        'brown',
#        'grey',
#        'green', #4
#        'orange', #5
#        'pink',
#        'purple',#7
#        'red', #8
#        'white', #9
#        'yellow']
#        
#        img_name = './datasets/heyrin/hey/12350_mask.jpg'
#        im_mask = Image.open(img_name)       
#        im_mask = im_mask.resize((64,64))
#        piltransforms = transforms.ToTensor()
#        torch_mask = piltransforms(im_mask)
#        mask = torch_mask[0]>0
#        
#        im_path = './datasets/apple2orange/apple2orange/testA\\n07740461_12350.jpg'
#        im = cv2.imread(im_path)
#        im = cv2.resize(im, (64,64))
#        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
#   
#        colormap = label_colors(im_rgb)
#    
#        with open('colormap_18.pickle', 'rb') as f:
#            dict_colormap = pickle.load(f)
#        
#        for i in range(output.shape[2]):
#            for j in range(output.shape[3]):
#                if mask[i,j]:
#                    print(mask[i,j])
#                    print(color_names[colormap[i,j]])
#                    output[0,:,i,j] = dict_colormap[color_names[colormap[i,j]]]+1
#        
#        return output
#    
#    def change_color(module, input, output):
#        color_names = [
#        'black', #0
#        'blue', #1
#        'brown',
#        'grey',
#        'green', #4
#        'orange', #5
#        'pink',
#        'purple',#7
#        'red', #8
#        'white', #9
#        'yellow']
#        
#        img_name = './datasets/heyrin/hey/12350_mask.jpg'
#        im_mask = Image.open(img_name)       
#        im_mask = im_mask.resize((64,64))
#        piltransforms = transforms.ToTensor()
#        torch_mask = piltransforms(im_mask)
#        mask = torch_mask[0]>0
#        
#        im_path = './datasets/apple2orange/apple2orange/testA\\n07740461_12350.jpg'
#        im = cv2.imread(im_path)
#        im = cv2.resize(im, (64,64))
#        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
#   
#        colormap = label_colors(im_rgb)
#    
#        with open('colormap_18.pickle', 'rb') as f:
#            dict_colormap = pickle.load(f)
#        
#        color_tensor = dict_colormap['black']
#        color_orange = dict_colormap['orange']
#        for i in range(output.shape[2]):
#            for j in range(output.shape[3]):
#                if mask[i,j]:
#                    output[0,color_tensor>2,i,j] = output[0].mean()+10
#                    output[0,color_orange>2,i,j] = output[0].min()
#                        #output[0,:,i,j] = dict_colormap[color_names[colormap[i,j]]]+1
#        
#        return output
        
    
    for idx, layer in enumerate(model.netG.model):
        if isinstance(layer, ResnetBlock):
            layer.register_forward_hook(get_activation('Resblock_'+str(idx)))
            if (idx==18):
                layer.register_forward_hook(change_color)  
            for idx_, layer_ in enumerate(layer.conv_block):
                if isinstance(layer_, nn.Conv2d) or isinstance(layer_, nn.ConvTranspose2d):
                    print(layer_)
                    layer_.register_forward_hook(get_activation(str(idx)+'_'+str(idx_)))
                    #if (str(idx)+'_'+str(idx_)) in ['10_5', '11_5', '12_5', '13_5', '14_5', '15_5','16_5', '17_5', '18_5' ]:
                    #    print('hook!')
                    #    layer_.register_forward_hook(off_all_activation)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            print(layer)
            layer.register_forward_hook(get_activation(str(idx)))
            #if (str(idx)) == '7':
            #    print('hook!')
            #    layer.register_forward_hook(change_color)               
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        activation = {}
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        file_name = os.path.join("./activations", str(img_path).split('\\n')[-1][:-6] +"_map.pickle")
        with open(file_name, 'wb') as file:
            pickle.dump(activation, file)
    #webpage.save()  # save the HTML
