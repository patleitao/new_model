from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
import os
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from utils import *
import time

from libs.pconv_model import PConvUnet


rec_model = PConvUnet(vgg_weights='pytorch_to_keras_vgg16.h5', inference_only=False)
rec_model.load(r'pconv_imagenet.h5', train_bn=False)

high_feat_model = VGG16(weights='imagenet', include_top=False)

base_model = VGG19(weights='imagenet')
feat0_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
feat1_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
feat2_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
feat3_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
feat4_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


imagelist = []
mode = 'RGB'
rootdir = '../data/ALLSTIMULI'
for subdir, dirs, files in os.walk(rootdir):
    print('here')
    for file in files:
        filename = os.path.basename(file)
        img = imread(subdir + '/' + filename, mode=mode)
        imagelist.append(img)

imagelist= np.array(imagelist)

popoutlist = []
mode = 'RGB'
rootdir = '../data/popout'
for subdir, dirs, files in os.walk(rootdir):
    print('here')
    for file in files:
        filename = os.path.basename(file)
        img = imread(subdir + '/' + filename, mode=mode)
        popoutlist.append(img)

popoutlist = np.array(popoutlist)

print('images loaded')

# outputs list of features from level1 to level5 of VGG16 hierarchy (with models as list of vgg layer outputs)

def get_img_features(img, mask_size, models):
    gt_features = []
    for model in models:
        x = np.array(img, copy=True)
        x = imresize(x, (224,224))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        gt_feature = model.predict(x)
        gt_features.append(gt_feature)
    stride = int(mask_size/2)
    features = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i in range(0+stride, 512-(stride*2), stride):
        for j in range(0+stride, 512-(stride*2), stride):
            mask = np.ones((512,512,3))
            mask[i:i+mask_size, j:j+mask_size] = 0
            masked_img = np.array(img, copy=True)
            masked_img = masked_img / 255.0
            masked_img[mask==0] = 1
            masked_img = np.expand_dims(masked_img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            new_img = rec_model.predict([masked_img, mask])

            new_img_mod = imresize(new_img[0], (224,224))
            new_img_mod = np.expand_dims(new_img_mod, axis=0)
            new_img_mod = preprocess_input(new_img_mod)

            for idx, model in enumerate(models):
                feature = model.predict(new_img_mod)
                features[idx].append(feature)

    feat_diff_map = {0: [], 1: [], 2: [], 3: [], 4: []}
    for level, feats in features:
        for feature in feats:
            value = np.mean(np.absolute(gt_features[level]-feature))
            feat_diff_map[level].append(value)
        feat_diff_map[level] = np.asarray(feat_diff_map[level])
        feat_diff_map[level] = feat_diff_map[level].reshape((8,8))

    return feat_diff_map



def get_img_features_no_rec(img, mask_size, model):
    x = np.array(img, copy=True)
    x = imresize(x, (224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    gt_feature = = model.predict(x)
    stride = int(mask_size/2)
    features = []
    for i in range(0+stride, 512-(stride*2), stride):
        for j in range(0+stride, 512-(stride*2), stride):
            mask = np.ones((512,512,3))
            mask[i:i+mask_size, j:j+mask_size] = 0
            masked_img = np.array(img, copy=True)
            masked_img = masked_img / 255.0
            masked_img[mask==0] = 1
            masked_img = np.expand_dims(masked_img, axis=0)
            new_img_mod = imresize(masked_img, (224,224))
            new_img_mod = np.expand_dims(new_img_mod, axis=0)
            new_img_mod = preprocess_input(new_img_mod)
            feature = model.predict(new_img_mod)
            features.append(feature)
    return gt_feature, features

def get_error_map(img, mask_size):
    stride = int(mask_size/2)
    error_values_map = np.zeros((8, 8))
    for i in range(0+stride, 512-(stride*2), stride):
        for j in range(0+stride, 512-(stride*2), stride):
            mask = np.ones((512,512,3))
            mask[i:i+mask_size, j:j+mask_size] = 0
            masked_img = np.array(img, copy=True)
            masked_img = masked_img / 255.0
            masked_img[mask==0] = 1
            masked_img = np.expand_dims(masked_img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            new_img = rec_model.predict([masked_img, mask])
            x = int(i/stride)
            y = int(j/stride)
            error_values_map[x, y] = np.sum(np.absolute(img[i:i+mask_size, j:j+mask_size, :] - new_img[0, i:i+mask_size, j:j+mask_size, :]))
    return error_values_map


def get_feat_diff_map(gt_feature, features):
    feat_diff_map = []
    for feature in features:
        value = np.mean(np.absolute(gt_feature-feature))
        feat_diff_map.append(value)
    feat_diff_map = np.asarray(feat_diff_map)
    feat_diff_map = feat_diff_map.reshape((8,8))
    return feat_diff_map



# Combine all to only do one reconstruction iteration

imagelist_true_maps = []
for idx, img in enumerate(imagelist):
    shape = img.shape
    img = imresize(img, (512,512))
    starttime = time.time()
    feat_diff_map = get_img_features(img, 96, [feat0_model, feat1_model, feat2_model, feat3_model, feat4_model])
    true_maps = {0: np.ones((10,10))*np.min(feat_diff_map[0]), 0: np.ones((10,10))*np.min(feat_diff_map[1]), 0: np.ones((10,10))*np.min(feat_diff_map[2]), 0: np.ones((10,10))*np.min(feat_diff_map[3]), 0: np.ones((10,10))*np.min(feat_diff_map[4])}
    for level, true_map in true_maps:
        true_map[1:9, 1:9] = feat_diff_map[level]
        true_map = imresize(true_map, shape, interp='bicubic')
        true_maps[level].append(true_map)
    endtime = time.time()
    duration = endtime - starttime
    print('this image took ' + duration)
    imagelist_true_maps.append(true_maps)

maps0 = []
maps1 = []
maps2 = []
maps3 = []
maps4 = []

for true_maps in imagelist_true_maps:
    maps0.append(true_maps[0])
    maps1.append(true_maps[1])
    maps2.append(true_maps[2])
    maps3.append(true_maps[3])
    maps4.append(true_maps[4])

maps0 = np.asarray(maps0)
maps1 = np.asarray(maps1)
maps2 = np.asarray(maps2)
maps3 = np.asarray(maps3)
maps4 = np.asarray(maps4)

save_array(maps0, 'sal_maps_0')
save_array(maps1, 'sal_maps_1')
save_array(maps1, 'sal_maps_2')
save_array(maps1, 'sal_maps_3')
save_array(maps1, 'sal_maps_4')



# # USE VGG16
# true_maps = []
# for idx, img in enumerate(imagelist):
#     shape = img.shape
#     img = imresize(img, (512,512))
#     starttime = time.time()
#     gt_feature, features = get_img_features(img, 96, high_feat_model)
#     feat_diff_map = get_feat_diff_map(gt_feature, features)
#     true_map = np.ones((10,10))*np.min(feat_diff_map)
#     true_map[1:9, 1:9] = feat_diff_map
#     true_map = imresize(true_map, shape, interp='bicubic')
#     true_maps.append(true_map)
#     endtime = time.time()
#     duration = endtime - starttime
#     print('this image took ' + duration)

# save_array(true_maps, 'true_maps_VGG16')


# print('high_feat results finished')

# # USE INTERMEDIATE LAYER OF VGG19
# true_maps = []
# for idx, img in enumerate(imagelist):
#     shape = img.shape
#     img = imresize(img, (512,512))
#     gt_feature, features = get_img_features(img, 96, low_feat_model)
#     feat_diff_map = get_feat_diff_map(gt_feature, features)
#     true_map = np.ones((10,10))*np.min(feat_diff_map)
#     true_map[1:9, 1:9] = feat_diff_map
#     true_map = imresize(true_map, shape, interp='bicubic')
#     true_maps.append(true_map)

# save_array(true_maps, 'true_maps_VGG19')

# print('low_feat results finished')


# VGG16 WITH NO RECONSTRUCTION
true_maps = []
for idx, img in enumerate(imagelist):
    shape = img.shape
    img = imresize(img, (512,512))
    gt_feature, features = get_img_features_no_rec(img, 96, high_feat_model)
    feat_diff_map = get_feat_diff_map(gt_feature, features)
    true_map = np.ones((10,10))*np.min(feat_diff_map)
    true_map[1:9, 1:9] = feat_diff_map
    true_map = imresize(true_map, shape, interp='bicubic')
    true_maps.append(true_map)

save_array(true_maps, 'true_maps_no_rec')


# ERROR FROM RECONSTRUCTION DIFFERENCE
true_maps = []
for idx, img in enumerate(imagelist):
    shape = img.shape
    img = imresize(img, (512,512))
    error_map = get_error_map(img, 96)
    true_map = np.ones((10,10))*np.min(error_map)
    true_map[1:9, 1:9] = error_map
    true_map = imresize(true_map, shape, interp='bicubic')
    true_maps.append(true_map)

save_array(true_maps, 'true_maps_error_rec')

