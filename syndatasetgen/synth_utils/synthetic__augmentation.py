"""synthetic__augmentation.py

Origin: ruth_archive2
"""
import cv2
import numpy as np
import torch

from ..synth_utils import augmentation

CROP_SIZE = (512, 512)
ROT_RANGE = [-180, 180]
GLOBAL_SCALE = 0.5
SCALE_RANGE = [0.8 * GLOBAL_SCALE, 1.25 * GLOBAL_SCALE]


def droplet_augmentation(image, num_droplets=3, radius_range=(5,30)):
    img_np = np.array(image.cpu(), dtype=float)
    img_np = np.transpose(img_np.squeeze(), (1,2,0))
    image_size = img_np.shape
    mask = np.zeros(img_np.shape, dtype=float)
    for i in range(num_droplets):
        this_centre = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
        this_radius = np.random.randint(radius_range[0], radius_range[1])
        mask = cv2.circle(mask, this_centre, this_radius, (1, 1, 1), -1)
    #
    blurred_image = cv2.GaussianBlur(img_np, (33, 33), 6)
    blurred_mask = cv2.GaussianBlur(mask, (101, 101), 0)
    blurred_section = cv2.multiply(blurred_mask, blurred_image)
    unblurred_section = cv2.multiply(1.0 - blurred_mask, img_np)
    result = cv2.add(blurred_section, unblurred_section)
    result = np.transpose(result, (2,0,1))[None,:]
    return torch.tensor(result, dtype=torch.float, device='cuda:0')

def augment_classifier(img,xf,label_img):
    """
    augment_classifier - performs random flips
    """
    size=(192, 192)
    size_np=np.array(size)
    centre_xf = augmentation.translation_matrices(-size_np[None, :] * 0.5)
    uncentre_xf = augmentation.translation_matrices(size_np[None, :] * 0.5)
    xy = np.ones((1,2))
    flip_xf=augmentation.scale_matrices(xy)
    flip_h = np.random.binomial(1, 0.5, size=(1,))
    if flip_h[0]:
      flip_xf[0, 0, 0] *= -1
    flip_v = np.random.binomial(1, 0.5, size=(1,))
    if flip_v[0]:
        flip_xf[0, 1, 1] *= -1
    xf = augmentation.cat_nx2x3(uncentre_xf,flip_xf,centre_xf)
    label_img = cv2.warpAffine(label_img, xf[0], size, flags=cv2.INTER_NEAREST)
    return img, xf, label_img

def augment_labels(img, xf, label_img, crop_size=CROP_SIZE, scale_range=SCALE_RANGE):
    crop_size_np = np.array(crop_size)
    crop_remainder = np.array(label_img.shape) - crop_size_np
    crop_offset = np.round(crop_remainder * np.random.uniform(0, 1, size=crop_remainder.shape)).astype(int)
    crop_xf = augmentation.translation_matrices(-crop_offset[None, ::-1])
    centre_xf = augmentation.translation_matrices(-crop_size_np[None, :] * 0.5)
    uncentre_xf = augmentation.translation_matrices(crop_size_np[None, :] * 0.5)
    rot_theta = np.random.uniform(np.radians(ROT_RANGE[0]), np.radians(ROT_RANGE[1]), size=(1,))
    rot_xf = augmentation.rotation_matrices(rot_theta)
    scale_factor = np.exp(np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]), size=(1,)))
    scale_xy = scale_factor[:, None] * np.ones((1, 2))
    scale_xf = augmentation.scale_matrices(scale_xy)
    flip_h = np.random.binomial(1, 0.5, size=(1,))
    if flip_h[0]:
        scale_xf[0, 0, 0] *= -1
    xf = augmentation.cat_nx2x3(uncentre_xf, scale_xf, rot_xf, centre_xf, crop_xf)
    label_img = cv2.warpAffine(label_img, xf[0], crop_size, flags=cv2.INTER_NEAREST)
    return img, xf, label_img


def augment_labels_and_scale(
        label_img,
        scale_range, 
        image_size, 
        position,
        rot_theta,
        flip):
    """
    mhf 11.12.24
    """

    position_np = np.array(position)
    rot_theta_np = np.array([rot_theta])
    

    # mhf BEWARE!!
    # :param xlats_xy: translations as an (N, 2) array (x,y)
    (y,x) = label_img.shape
    img_size_np = np.array((x,y))
    centre_xf = augmentation.translation_matrices(-img_size_np[None, :] * 0.5)

    uncentre_xf = augmentation.translation_matrices(position_np[None, :])


    # rotation
    rot_xf = augmentation.rotation_matrices(rot_theta_np)
    
    # scale
    scale_factor = np.exp(np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]), size=(1,)))
    scale_xy = scale_factor[:, None] * np.ones((1, 2))
    scale_xf = augmentation.scale_matrices(scale_xy)
    

    # flip
    if flip:
        flip_h = np.random.binomial(1, 0.5, size=(1,))
        if flip_h[0]:
            scale_xf[0, 0, 0] *= -1

           
    xf = augmentation.cat_nx2x3(uncentre_xf, scale_xf, rot_xf, centre_xf)
    label_img = cv2.warpAffine(label_img, xf[0], image_size, flags=cv2.INTER_NEAREST)

    return xf, label_img, scale_factor


def augment_labels_flip_scale_rotate(label_img, 
                                     xf,
                                     flip,
                                     scale_factor,
                                     rot_theta):
    """
    mhf 09.12.24 Similar to above but fixes some bugs.
    Allows scale, rotate, flip to be controlled by params
    """
    img_size_np = np.array(label_img.shape)
    centre_xf = augmentation.translation_matrices(-img_size_np[None, :] * 0.5)
    uncentre_xf = augmentation.translation_matrices(img_size_np[None, :] * 0.5)

    rot_theta = np.array([rot_theta])
    rot_xf = augmentation.rotation_matrices(rot_theta)

    scale_factor = np.array([scale_factor])
    scale_xy = scale_factor[:, None] * np.ones((1, 2))
    scale_xf = augmentation.scale_matrices(scale_xy)

    if flip:
        flip_h = np.random.binomial(1, 0.5, size=(1,))
        if flip_h[0]:
            scale_xf[0, 0, 0] *= -1

    xf = augmentation.cat_nx2x3(uncentre_xf, scale_xf, rot_xf, centre_xf)
    label_img = cv2.warpAffine(label_img, xf[0], CROP_SIZE, flags=cv2.INTER_NEAREST)
    #
    return xf, label_img


def augmentation_crop_flip_scale_rotate(img, xf, label_img):
    crop_size_np = np.array(CROP_SIZE)
    crop_remainder = np.array(label_img.shape) - crop_size_np
    crop_offset = np.round(crop_remainder * np.random.uniform(0, 1, size=crop_remainder.shape)).astype(int)
    crop_xf = augmentation.translation_matrices(-crop_offset[None, ::-1])
    centre_xf = augmentation.translation_matrices(-crop_size_np[None, :] * 0.5)
    uncentre_xf = augmentation.translation_matrices(crop_size_np[None, :] * 0.5)
    rot_xf = build_rotation_xf(ROT_RANGE)
    scale_xf = build_scale_xf(SCALE_RANGE, flip=True)
    xf = augmentation.cat_nx2x3(uncentre_xf, scale_xf, rot_xf, centre_xf, crop_xf)
    label_img = cv2.warpAffine(label_img, xf[0], CROP_SIZE, flags=cv2.INTER_NEAREST)
    #
    return img, xf, label_img


def build_rotation_xf(rotation_range):
    rot_theta = np.random.uniform(np.radians(rotation_range[0]), np.radians(rotation_range[1]), size=(1,))
    rot_xf = augmentation.rotation_matrices(rot_theta)
    return rot_xf

def build_scale_xf(scale_range, flip=False):
    scale_factor = np.exp(np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]), size=(1,)))
    scale_xy = scale_factor[:, None] * np.ones((1, 2))
    scale_xf = augmentation.scale_matrices(scale_xy)
    if flip:
        flip_h = np.random.binomial(1, 0.5, size=(1,))
        if flip_h[0]:
            scale_xf[0, 0, 0] *= -1
    return scale_xf, scale_factor


def augmentation_scale(img, xf, label_img):
    scale_xf = build_scale_xf([0.5, 0.5])
    xf = augmentation.cat_nx2x3(scale_xf)
    label_img = cv2.warpAffine(label_img, xf[0], CROP_SIZE, flags=cv2.INTER_NEAREST)
    #
    return img, xf, label_img
