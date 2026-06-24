import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def join_xf_col(xf, col_mtx, col_off):
    return np.concatenate([xf.reshape((-1, 6)),
                           col_mtx.reshape((-1, 9)),
                           col_off.reshape((-1, 3))], axis=1).astype(np.float32)


def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3_2(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def cat_nx2x3(*x):
    """
    Multiply the N 2x3 transformations stored in the arrays in `x`
    :param x: transformation matrices, tuple of (N,2,3) arrays
    :return: `x[0] . x[1] . ... . x[N-1]`
    """
    y = x[0]
    for i in range(1, len(x)):
        y = cat_nx2x3_2(y, x[i])
    return y

def translation_matrices(xlats_xy):
    """
    Generate translation matrices
    :param xlats_xy: translations as an (N, 2) array (x,y)
    :return: translations matrices, (N,2,3) array
    """
    N = len(xlats_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    xf[:, :, 2] = xlats_xy
    return xf

def apply_translations_to_matrices(xf, xlats_xy):
    """
    Apply translation to existing matrices
    :param xf: matrices to apply translations to
    :param xlats_xy: translations as an (N, 2) array (x,y)
    :return: matrices with translations, (N,2,3) array
    """
    xf = xf.copy()
    xf[:, :, 2] += xlats_xy
    return xf

def scale_matrices(scale_xy):
    """
    Generate rotation matrices
    :param scale_xy: X and Y scale factors as an (N, 2) array (x,y)
    :return: scale matrices, (N,2,3) array
    """
    N = len(scale_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = scale_xy[:, 0]
    xf[:, 1, 1] = scale_xy[:, 1]
    return xf

def rotation_matrices(thetas):
    """
    Generate rotation matrices
    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = len(thetas)
    c = np.cos(thetas)
    s = np.sin(thetas)
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = c
    rot_xf[:, 1, 0] = s
    rot_xf[:, 0, 1] = -s
    return rot_xf

def grid_to_image_size_nx2x3(N, size):
    """
    Generate matrices that will transform a [-1, 1] grid to the size of the specified image, so that
    transformations will operate on a co-ordinate frame that is the size of the image and centred
    at the image centre. This way rotations will not result in shearing and translations will be
    performed in pixel co-ordinates.

    Two matrix arrays are returned; the grid-to-image matrices and their inverses.

    :param N: the number of matrices to generate
    :param size: the size of the image being transformed as a `(height, with)` tuple
    :return: tuple of `(mtx, mtx_inv)` where `mtx` and `mtx_inv` are the matrices and their inverses
    respectively. `mtx` and `mtx_inv` are NumPy arrays of shape `(N,2,3)`
    """
    x = float(size[1]) / 2.0
    y = float(size[0]) / 2.0

    mtx = identity_xf(N)
    mtx[:, 0, 0] = x
    mtx[:, 1, 1] = y

    mtx_inv = identity_xf(N)
    mtx_inv[:, 0, 0] = 1.0 / x
    mtx_inv[:, 1, 1] = 1.0 / y

    return mtx, mtx_inv

def grid_to_px_nx2x3(N, src_size, tgt_size=None):
    """
    Generate matrices that will transform a [-1, 1] grid to an image of size `size`.

    :param N: the number of matrices to generate
    :param tgt_size: the size of the output image as a `(height, with)` tuple
    :param src_size: the size of the image being transformed as a `(height, with)` tuple
    :return: tuple of `(grid_to_tgt_px, src_px_to_grid)` where:
        `grid_to_tgt_px` transforms a [-1,1] grid to a `tgt_size` pixel image
        `src_px_to_grid` transforms a `src_size` pixel image to a [-1, 1] grid
    `grid_to_tgt_px` and `src_px_to_grid` are NumPy arrays of shape `(N,2,3)`
    """
    src_x = float(src_size[1]-1) / 2.0
    src_y = float(src_size[0]-1) / 2.0

    if tgt_size is not None:
        tgt_x = float(tgt_size[1]-1) / 2.0
        tgt_y = float(tgt_size[0]-1) / 2.0
    else:
        tgt_x = src_x
        tgt_y = src_y

    grid_to_tgt_px = identity_xf(N)
    grid_to_tgt_px[:, 0, 0] = tgt_x
    grid_to_tgt_px[:, 1, 1] = tgt_y
    grid_to_tgt_px[:, 0, 2] = tgt_x
    grid_to_tgt_px[:, 1, 2] = tgt_y

    src_px_to_grid = identity_xf(N)
    src_px_to_grid[:, 0, 0] = 1.0 / src_x
    src_px_to_grid[:, 1, 1] = 1.0 / src_y
    src_px_to_grid[:, 0, 2] = -1.0
    src_px_to_grid[:, 1, 2] = -1.0

    return grid_to_tgt_px, src_px_to_grid


def open_cv_grid_mapping_to_torch(mapping_y, mapping_x, src_image_shape):
    """
    Convert a mapping used by `cv2.remap` for use by `torch.nn.function.grid_sample`

    :param mapping_y: y-co-ordinate of mapping in range [-0.5: image_height-0.5]
    :param mapping_x: y-co-ordinate of mapping in range [-0.5: image_width-0.5]
    :param src_image_shape: the shape of the source image that is being mapped
    :return: numpy array of shape (1, height, width, 2)
    """
    mapping_y_01 = ((mapping_y + 0.5 - src_image_shape[0] * 0.5) / (src_image_shape[0] * 0.5 - 0.5))
    mapping_x_01 = ((mapping_x + 0.5 - src_image_shape[1] * 0.5) / (src_image_shape[1] * 0.5 - 0.5))
    return np.stack([mapping_x_01, mapping_y_01], axis=2)[None, ...]


def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


def identity_col_mtx(N):
    """
    Construct N identity 3x3 colour matrices
    :return: array of shape (N, 3, 3)
    """
    col_mtx = np.zeros((N, 3, 3), dtype=np.float32)
    col_mtx[:, 0, 0] = col_mtx[:, 1, 1] = col_mtx[:, 2, 2] = 1.0
    return col_mtx


def identity_col_off(N):
    """
    Construct N identity colour offsets
    :return: array of shape (N, 3)
    """
    col_off = np.zeros((N, 3), dtype=np.float32)
    return col_off


def identity_xf_col(N):
    xf = identity_xf(N)
    col_mtx = identity_col_mtx(N)
    col_off = identity_col_off(N)
    return join_xf_col(xf, col_mtx, col_off)


def apply_grid_to_image(mtx, size):
    """
    Apply apsect ratio correction (see `aspect_nx2x3` function) for images of size `size` to the matrices in `mtx`

    :param mtx: transformation matrices as a `(N,2,3)` arrays
    :param size: image size as a `(height, width)` tuple
    :return: corrected matrices as a `(N,2,3)` array
    """
    asp, asp_inv = grid_to_image_size_nx2x3(len(mtx), size)
    return cat_nx2x3(asp_inv, cat_nx2x3(mtx, asp))


def axis_angle_rotation_matrices(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    :param axis: the axes
    :param theta: the angles in radians
    :return: matrices as an (N,3,3) array
    """
    # Normalize
    axis = axis/np.sqrt(np.sum(axis*axis, axis=1, keepdims=True))
    a = np.cos(theta/2)
    axis_sin_theta = -axis*np.sin(theta/2)[:, None]
    b = axis_sin_theta[:, 0]
    c = axis_sin_theta[:, 1]
    d = axis_sin_theta[:, 2]
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot.transpose(2, 0, 1)


def axis_to_axis_rotation_matrices(axes_a, axes_b, normalize=False, epsilon=1e-12):
    """
    Return the rotation matrices that rotate the axes in `axes_a` onto the axes in `axes_b`

    :param axes_a: source axes
    :param axes_b: target axes
    :param normalize: if True, normalize axes_a and axes_b before use
    :param epsilon: if normalize is True, use this epsilon value to prevent divide by 0
    :return: matrices as an (N,3,3) array
    """
    if normalize:
        axes_a = axes_a / np.maximum(np.sqrt((axes_a * axes_a).sum(axis=1, keepdims=True)), epsilon)
        axes_b = axes_b / np.maximum(np.sqrt((axes_b * axes_b).sum(axis=1, keepdims=True)), epsilon)
    rotation_axes = np.cross(axes_a, axes_b)
    rotation_cos_thetas = (axes_a * axes_b).sum(axis=1)

    # Detect cases where axes are perfectly aligned or opposed
    aligned_or_opposed = (rotation_cos_thetas > (1.0 - epsilon)) | (rotation_cos_thetas < (-1.0 + epsilon))
    if aligned_or_opposed.any():
        # Choose most perpendicular base axis to A
        most_perp_to_a = (np.argmin(abs(axes_a), axis=1)[:, None] == np.array([[0, 1, 2]])).astype(float)
        # Get axis perpendicular to both
        perp_rot_axis = np.cross(axes_a, most_perp_to_a)
        # Normalize
        perp_to_a = perp_rot_axis / np.maximum(np.sqrt((perp_rot_axis * perp_rot_axis).sum(axis=1, keepdims=True)), epsilon)
        # Replace rotation axes with valid axes as the result of the initial cross product will be degenerate
        rotation_axes[aligned_or_opposed] = perp_to_a[aligned_or_opposed]
    rotation_thetas = np.arccos(np.clip(rotation_cos_thetas, -1.0, 1.0))
    return axis_angle_rotation_matrices(rotation_axes, rotation_thetas)


def triangle_to_triangle_matrices(src_tris_xy, tgt_tris_xy):
    """
    Return transformation matrices that transform the triangles in `src_tris_xy` to the triangles in `tgt_tris_xy`
    :param src_tris_xy: source triangles as a (N, 3, 2) array, where co-ordinates are xy
    :param tgt_tris_xy: target triangles as a (N, 3, 2) array, where co-ordinates are xy
    :return: transformation matrices as an (N, 3, 2) array
    """
    if len(src_tris_xy) != len(tgt_tris_xy) and len(src_tris_xy) != 1 and len(tgt_tris_xy) != 1:
        raise ValueError('Number of source tris ({}) != number of target tris ({})'.format(len(src_tris_xy), len(tgt_tris_xy)))

    tgt_u = tgt_tris_xy[:, 1, :] - tgt_tris_xy[:, 0, :]
    tgt_v = tgt_tris_xy[:, 2, :] - tgt_tris_xy[:, 0, :]
    src_u = src_tris_xy[:, 1, :] - src_tris_xy[:, 0, :]
    src_v = src_tris_xy[:, 2, :] - src_tris_xy[:, 0, :]

    # Target matrix
    # [u[0], v[0], a[0]]
    # [u[1], v[1], a[1]]
    tgt_mtx = np.stack([tgt_u, tgt_v, tgt_tris_xy[:, 0, :]], axis=2)

    # Source matrix
    # [u[0], v[0], a[0]]
    # [u[1], v[1], a[1]]
    src_mtx = np.stack([src_u, src_v, src_tris_xy[:, 0, :]], axis=2)

    src_to_tgt_matrix = cat_nx2x3_2(tgt_mtx, inv_nx2x3(src_mtx))

    return src_to_tgt_matrix


def compute_transformed_image_padding(image_size, xf):
    """
    Compute the padding required to completely contain an image of size `image_size` transformed by the transformations
    in `xf`

    :param image_size: image size as a tuple `(height, width)`
    :param xf: transformations as an `(N, 2, 3)` array
    :return: tuple `(padding, padded_size)` where `padding` is an `(N, 4)` array where each row is
            [pad_top, pad_bottom, pad_left, pad_right] and `padded_size` is an `(N, 2)` array where each
            row is [padded_height, padded_width]
    """
    if len(xf) != 1:
        raise NotImplementedError('Check that this works with batch size > 1')
    height, width = image_size

    xf_corners = centre_xf(xf, image_size)

    # (4, 2) array where each row is a corner; note co-ordinates used for transformations are [x,y] NOT [y,x]
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    corners_z = np.matmul(xf_corners[:, :, :2], corners.T).transpose(0, 2, 1) + xf_corners[:, :, 2][: None, :]

    left_top = np.floor(corners_z.min(axis=1)).astype(int)
    right_bottom = np.ceil(corners_z.max(axis=1)).astype(int)

    pad_lower = -left_top
    pad_upper = right_bottom - np.array([[width, height]])

    padding = np.stack([pad_lower[:, 1], pad_upper[:, 1], pad_lower[:, 0], pad_upper[:, 0]], axis=1)
    padded_size = right_bottom - left_top

    return padding, padded_size[:, ::-1]


def complete_image_transform(image_size, xf, padding, crop_topleft, xlat):
    if len(xf) != 1:
        raise NotImplementedError('Batch size != 1 not supported')

    # Compute padded size
    height, width = image_size
    pad_top, pad_bottom, pad_left, pad_right = padding
    pad_height = height + pad_top + pad_bottom
    pad_width = width + pad_left + pad_right
    padded_size = np.array([int(pad_height), int(pad_width)])

    # Compute padding scale transformation that scales the grid so that it fits the original image
    pad_scale_x = float(pad_width - 1) / float(width - 1)
    pad_scale_y = float(pad_height - 1) / float(height - 1)
    pad_scale_xf = np.array([[[1.0 / pad_scale_x, 0.0, 0.0], [0.0, 1.0 / pad_scale_y, 0.0]]])
    pad_scale_xf = np.repeat(pad_scale_xf, len(xf), axis=0).astype(np.float32)

    xf_xlat = np.array([[[1.0, 0.0, xlat[1] - crop_topleft[1]], [0.0, 1.0, xlat[0] - crop_topleft[0]]]])

    # Padded transformation is the concatentation of the padding scale, the translation and the transformation
    xf_padded_cropped_xlat = cat_nx2x3(pad_scale_xf, xf_xlat, xf)

    return xf_padded_cropped_xlat, padded_size


def transform_points(xf, points_xy):
    """
    Apply the transformation matrix `xf` to the points in `points`.

    :param xf: transformation as a (2, 3) NumPy array
    :param points_xy: points as a (N, 2) array where each point is of the form (x, y), not (y, x)
    :return: (N, 2) array
    """
    return np.tensordot(xf[:2, :2], points_xy, [[1], [1]]).T + xf[:, 2][None, :]



def compute_upscale_and_downscale_factor(sample_name, cell_size_dist_db, target_cell_size, image_size):
    upscale = downscale = 1
    if cell_size_dist_db is not None:
        dist = cell_size_dist_db['name_to_size'][sample_name]
        mean = dist['mean']
        p25 = dist['p25']
        p75 = dist['p75']
        if p75 < target_cell_size[0]:
            upscale = int(round(float(target_cell_size[1]) / float(mean)))
            while (image_size[0] * image_size[1]) * (upscale * upscale) > (1280*1280) and upscale > 1:
                upscale -= 1
        elif p25 > target_cell_size[2]:
            downscale = int(round(float(mean) / float(target_cell_size[1])))
    return upscale, downscale



def torch_reflect_grid(x):
    return torch.abs(torch.fmod(torch.abs(x-1), 4.0)-2)-1

def torch_replicate_grid(x):
    return torch.clamp(x, -1.0, 1.0)

def torch_pad_grid(x, pad_mode):
    if pad_mode == 'reflect':
        return torch_reflect_grid(x)
    elif pad_mode == 'replicate':
        return torch_replicate_grid(x)
    elif pad_mode == 'constant':
        return x
    else:
        raise ValueError('Unknown pad_mode \'{}\'; should be reflect, replicate or constant'.format(pad_mode))


def torch_grid_sample_nearest(input, grid):
    scale_factor = (np.array([input.size(3), input.size(2)])-1)
    sf_var = torch.autograd.Variable(
        torch.from_numpy(scale_factor[None, None, None, :]).float().cuda())
    grid_n = (((grid + 1) * 0.5 * sf_var).round()) / sf_var * 2.0 - 1.0
    return F.grid_sample(input, grid_n, 'nearest')

def torch_grid_sample_nearest_cpu(input, grid):
    scale_factor = (np.array([input.size(3), input.size(2)])-1)
    sf_var = torch.autograd.Variable(
        torch.from_numpy(scale_factor[None, None, None, :]).float())
    grid_n = (((grid + 1) * 0.5 * sf_var).round()) / sf_var * 2.0 - 1.0
    return F.grid_sample(input, grid_n, 'nearest')


def torch_scaled_grid(xform, size, scale_factor):
    grid_scale = np.array([float(size[2]-scale_factor) / float(size[2]-1),
                           float(size[3]-scale_factor) / float(size[3]-1)])
    xf_scale = np.zeros_like(xform)
    xf_scale[:, 0, 0] = grid_scale[0]
    xf_scale[:, 1, 1] = grid_scale[1]
    xf = cat_nx2x3(xform, xf_scale)
    xf_var = torch.autograd.Variable(torch.from_numpy(xf).cuda())
    grid = F.affine_grid(xf_var, torch.Size((size[0], size[1], size[2] // scale_factor, size[3] // scale_factor)))
    return grid


def torch_scaled_grid_cpu(xform, size, scale_factor):
    grid_scale = np.array([float(size[2]-scale_factor) / float(size[2]-1),
                           float(size[3]-scale_factor) / float(size[3]-1)])
    xf_scale = np.zeros_like(xform)
    xf_scale[:, 0, 0] = grid_scale[0]
    xf_scale[:, 1, 1] = grid_scale[1]
    xf = cat_nx2x3(xform, xf_scale)
    xf_var = torch.autograd.Variable(torch.from_numpy(xf))
    grid = F.affine_grid(xf_var, torch.Size((size[0], size[1], size[2] // scale_factor, size[3] // scale_factor)))
    return grid



def cv_to_torch(mtx, dst_size, src_size=None):
    """
    Convert transformation matrix that can be used by `cv2.warpAffine` to work with PyTorch grid sampleing

    :param mtx: transformation matrices as a `(N,2,3)` arrays
    :param size: image size as a `(height, width)` tuple
    :return: corrected matrices as a `(N,2,3)` array
    """
    dst_scale_y = float(dst_size[0] - 1) / 2.0
    dst_scale_x = float(dst_size[1] - 1) / 2.0

    if src_size is None:
        src_scale_y = dst_scale_y
        src_scale_x = dst_scale_x
    else:
        src_scale_y = float(src_size[0] - 1) / 2.0
        src_scale_x = float(src_size[1] - 1) / 2.0


    N = len(mtx)

    # OpenCV transformations modify the image grid, whereas PyTorch sampling uses sample points, so
    # we must invert
    mtx = inv_nx2x3(mtx)

    # Matrix to transform torch grid [-1 to 1] co-ordinates to OpenCV image space
    torch_cv = identity_xf(N)
    torch_cv[:, 0, 0] = dst_scale_x
    torch_cv[:, 1, 1] = dst_scale_y
    torch_cv[:, 0, 2] = dst_scale_x
    torch_cv[:, 1, 2] = dst_scale_y

    # Matrix to transform from OpenCV image space back to torch grid
    cv_torch = identity_xf(N)
    cv_torch[:, 0, 0] = 1.0 / src_scale_x
    cv_torch[:, 1, 1] = 1.0 / src_scale_y
    cv_torch[:, 0, 2] = -1.0
    cv_torch[:, 1, 2] = -1.0

    return cat_nx2x3(cv_torch, cat_nx2x3(mtx, torch_cv))



def torch_matmul_imagetensor_colour_matrix(X, colour_matrix):
    X_out = []
    for i in range(X.size()[0]):
        img = X[i, :, :, :].permute(1, 2, 0)
        img_m = torch.matmul(img, colour_matrix[i, :, :]).permute(2, 0, 1)
        X_out.append(img_m[None, ...])

    X_out = torch.cat(X_out)

    return X_out




def torch_flip_tensor(x, dim):
    size_in_dim = x.size()[dim]
    indices = torch.arange(size_in_dim - 1, -1, -1).long()
    if x.is_cuda:
        indices = indices.cuda()
    return x.index_select(dim, indices)


def torch_flip_var(x, dim):
    size_in_dim = x.size()[dim]
    indices = torch.arange(size_in_dim - 1, -1, -1).long()
    if x.is_cuda:
        indices = indices.cuda()
    indices = torch.autograd.Variable(indices)
    return x.index_select(dim, indices)



def _scale_factor(scale_u_range, scale_x_range, scale_y_range, rng):
    """
    Compute a random scale factor, given scale ranges for uniform, X and Y scales

    :param scale_u_range: tuple `(lower_range, upper_range)` or None
    :param scale_x_range: tuple `(lower_range, upper_range)` or None
    :param scale_y_range: tuple `(lower_range, upper_range)` or None
    :param rng: random number generator
    :return: scale factor as a (2,) NumPy array or None for no scale
    """
    if scale_u_range is not None or scale_x_range is not None or scale_y_range is not None:
        scale_factor = np.array([1.0, 1.0])
        if scale_u_range is not None:
            scale_factor *= rng.uniform(low=scale_u_range[0], high=scale_u_range[1], size=(1,))[0]
        if scale_x_range is not None:
            scale_factor[1] *= rng.uniform(low=scale_x_range[0], high=scale_x_range[1], size=(1,))[0]
        if scale_y_range is not None:
            scale_factor[0] *= rng.uniform(low=scale_y_range[0], high=scale_y_range[1], size=(1,))[0]

        return scale_factor
    else:
        return None

def _relative_crop_centre(image_size, crop_size, rng):
    """
    Randomly draw a crop position, given the size of the image from which the crop is to be drawn and the crop size.
    Returns the position of the crop centre relative to the image centre

    :param image_size: size of the image from which we are cropping a region as a (2,) NumPy array
    :param crop_size: size of the region that we are cropping from the image as a (2,) NumPy array
    :param rng: random number generator
    :return: Crop centre relative to image centre as a (2,) NumPy array
    """
    extra = np.maximum(image_size - crop_size, 0.0)
    u = rng.uniform(-0.5, 0.5, size=(2,))
    return u * extra

def _crop_adjustment(adjust_range, scale_factor, rng):
    """
    Randomly draw a crop adjustment that is within the range [-adjust_range*0.5, adjust_range*0.5].
    The adjustment range is in the co-ordinate frame of the crop, so if `scale_factor` is not None,
    the adjustment range is divided by the scale factor prior to drawing the adjustment

    :param adjust_range: the size of the adjustment range
    :param scale_factor: scale factor applied to crop
    :param rng: random number generator
    :return: crop adjustment as a (2,) NumPy array
    """
    adjust_range = np.array([adjust_range, adjust_range])
    if scale_factor is not None:
        adjust_range = adjust_range / scale_factor
    u = rng.uniform(-0.5, 0.5, size=(2,))
    return u * adjust_range

def _crop_and_pad(X, labels, rel_crop_centre, in_region_shape):
    """

    :param X:
    :param rel_crop_centre:
    :param in_region_shape:
    :return:
    """
    crop_centre = np.array(X.shape[:2]) * 0.5 + rel_crop_centre
    in_crop_lower = np.round(crop_centre - in_region_shape * 0.5).astype(int)
    in_crop_upper = np.round(crop_centre + in_region_shape * 0.5).astype(int)

    in_crop_lower = int(in_crop_lower[0]), int(in_crop_lower[1])
    in_crop_upper = int(in_crop_upper[0]), int(in_crop_upper[1])

    crop_y = max(in_crop_lower[0], 0), min(in_crop_upper[0], X.shape[0])
    crop_x = max(in_crop_lower[1], 0), min(in_crop_upper[1], X.shape[1])

    pad_y = max(-in_crop_lower[0], 0), max(in_crop_upper[0] - X.shape[0], 0)
    pad_x = max(-in_crop_lower[1], 0), max(in_crop_upper[1] - X.shape[1], 0)

    X = X[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1], :]
    if labels is not None:
        labels = labels[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    X = np.pad(X, [pad_y, pad_x, (0, 0)], mode='constant')
    if labels is not None:
        labels = np.pad(labels, [pad_y, pad_x], mode='constant')

    return X, labels, -np.array(in_crop_lower)



def augment_image_twice(X, fixed_shape, crop_border, constrain_crop, hflip, vflip, hvflip, constrain_flip,
                        scale_u_range, scale_x_range, scale_y_range, constrain_scale, rng=None):
    if rng is None:
        rng = np.random

    in_region_shape_a = in_region_shape_b = np.array(fixed_shape)
    image_shape = np.array(X.shape[:2])

    scale_factor_a = _scale_factor(scale_u_range, scale_x_range, scale_y_range, rng)
    if constrain_scale:
        scale_factor_b = scale_factor_a
    else:
        scale_factor_b = _scale_factor(scale_u_range, scale_x_range, scale_y_range, rng)
    if scale_factor_a is not None:
        in_region_shape_a = in_region_shape_a / scale_factor_a
    if scale_factor_b is not None:
        in_region_shape_b = in_region_shape_b / scale_factor_b

    rel_crop_centre_both = _relative_crop_centre(image_shape, np.maximum(in_region_shape_a, in_region_shape_b), rng)
    if constrain_crop:
        adj = _crop_adjustment(crop_border, scale_factor_a, rng)
        rel_crop_centre_a = rel_crop_centre_both + adj
        rel_crop_centre_b = rel_crop_centre_both + adj
    else:
        rel_crop_centre_a = rel_crop_centre_both + _crop_adjustment(crop_border, scale_factor_a, rng)
        rel_crop_centre_b = rel_crop_centre_both + _crop_adjustment(crop_border, scale_factor_b, rng)

    cropped_X_a, _, xlat_yx_a = _crop_and_pad(X, None, rel_crop_centre_a, in_region_shape_a)
    cropped_X_b, _, xlat_yx_b = _crop_and_pad(X, None, rel_crop_centre_b, in_region_shape_b)

    xf_a = translation_matrices(xlat_yx_a[None, ::-1])
    xf_b = translation_matrices(xlat_yx_b[None, ::-1])
    xf_flip_a = identity_xf(1)
    xf_flip_b = identity_xf(1)

    if cropped_X_a.shape != fixed_shape:
        # Scale if necessary
        scale_applied_xy_a = np.array(fixed_shape[::-1]).astype(float) / np.array(cropped_X_a.shape[:2][::-1])
        cropped_X_a = cv2.resize(cropped_X_a, fixed_shape[::-1], interpolation=cv2.INTER_LINEAR)
        xf_a = cat_nx2x3_2(scale_matrices(scale_applied_xy_a[None, :]), xf_a)

    if cropped_X_b.shape != fixed_shape:
        # Scale if necessary
        scale_applied_xy_b = np.array(fixed_shape[::-1]).astype(float) / np.array(cropped_X_b.shape[:2][::-1])
        cropped_X_b = cv2.resize(cropped_X_b, fixed_shape[::-1], interpolation=cv2.INTER_LINEAR)
        xf_b = cat_nx2x3_2(scale_matrices(scale_applied_xy_b[None, :]), xf_b)

    if hflip or vflip or hvflip:
        flip_flags_a = np.random.binomial(1, 0.5, 3)
        if constrain_flip:
            flip_flags_b = flip_flags_a
        else:
            flip_flags_b = np.random.binomial(1, 0.5, 3)

        if hflip & flip_flags_a[0]:
            cropped_X_a = cropped_X_a[:, ::-1, :]
            xf_a = cat_nx2x3(
                translation_matrices(np.array([[fixed_shape[1], 0.0]])),
                scale_matrices(np.array([[-1.0, 1.0]])),
                xf_a)
            xf_flip_a = cat_nx2x3(scale_matrices(np.array([[-1.0, 1.0]])), xf_flip_a)
        if hflip & flip_flags_b[0]:
            cropped_X_b = cropped_X_b[:, ::-1, :]
            xf_b = cat_nx2x3(
                translation_matrices(np.array([[fixed_shape[1], 0.0]])),
                scale_matrices(np.array([[-1.0, 1.0]])),
                xf_b)
            xf_flip_b = cat_nx2x3(scale_matrices(np.array([[-1.0, 1.0]])), xf_flip_b)

        if vflip & flip_flags_a[1]:
            cropped_X_a = cropped_X_a[::-1, :, :]
            xf_a = cat_nx2x3(
                translation_matrices(np.array([[0.0, fixed_shape[0]]])),
                scale_matrices(np.array([[1.0, -1.0]])),
                xf_a)
            xf_flip_a = cat_nx2x3(scale_matrices(np.array([[1.0, -1.0]])), xf_flip_a)
        if vflip & flip_flags_b[1]:
            cropped_X_b = cropped_X_b[::-1, :, :]
            xf_b = cat_nx2x3(translation_matrices(np.array([[0.0, fixed_shape[0]]])),
                             scale_matrices(np.array([[1.0, -1.0]])),
                             xf_b)
            xf_flip_b = cat_nx2x3(scale_matrices(np.array([[1.0, -1.0]])), xf_flip_b)

        if hvflip & flip_flags_a[2]:
            cropped_X_a = cropped_X_a.transpose(1, 0, 2)
            xf_a = cat_nx2x3(np.array([[[0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0]]]),
                             xf_a)
            xf_flip_a = cat_nx2x3(np.array([[[0.0, 1.0, 0.0],
                                             [1.0, 0.0, 0.0]]]),
                                  xf_flip_a)
        if hvflip & flip_flags_b[2]:
            cropped_X_b = cropped_X_b.transpose(1, 0, 2)
            xf_b = cat_nx2x3(np.array([[[0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0]]]),
                             xf_b)
            xf_flip_b = cat_nx2x3(np.array([[[0.0, 1.0, 0.0],
                                             [1.0, 0.0, 0.0]]]),
                                  xf_flip_b)

    return (cropped_X_a, xf_a, xf_flip_a), (cropped_X_b, xf_b, xf_flip_b)



class ImageAugmentation (object):
    def __init__(self, hflip=False, vflip=False, hvflip=False, xlat_range=0.0, affine_std=0.0, rot_range_mag=0.0,
                 light_scl_std=0.0, light_off_std=0.0,
                 scale_u_range=None, scale_x_range=None, scale_y_range=None,
                 scaled_rotation=False):
        self.hflip = hflip
        self.vflip = vflip
        self.hvflip = hvflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.rot_range_mag = rot_range_mag
        self.light_scl_std = light_scl_std
        self.log_light_scl_std = math.log(light_scl_std + 1.0)
        self.light_off_std = light_off_std
        self.scale_u_range = scale_u_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.scaled_rotation = scaled_rotation


    def aug_xforms(self, N, image_size, rng=None):
        if rng is None:
            rng = np.random
        xf = identity_xf(N)

        if self.hflip:
            x_hflip = rng.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 0, 0] = x_hflip.astype(np.float32)

        if self.vflip:
            x_vflip = rng.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 1, 1] = x_vflip.astype(np.float32)

        if self.hvflip:
            x_hvflip = rng.binomial(1, 0.5, size=(N,)) != 0
            xf[x_hvflip, :, :] = xf[x_hvflip, ::-1, :]

        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scl = rng.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(N,))
            xf[:, 0, 0] *= scl
            xf[:, 1, 1] *= scl
        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= rng.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(N,))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= rng.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(N,))

        if self.affine_std > 0.0:
            xf[:, :, :2] += rng.normal(scale=self.affine_std, size=(N, 2, 2))

        xlat_y_bounds = self.xlat_range * 2.0 / float(image_size[0])
        xlat_x_bounds = self.xlat_range * 2.0 / float(image_size[1])
        xf[:, 0, 2] += rng.uniform(low=-xlat_x_bounds, high=xlat_x_bounds, size=(N,))
        xf[:, 1, 2] += rng.uniform(low=-xlat_y_bounds, high=xlat_y_bounds, size=(N,))

        if self.rot_range_mag > 0.0:
            thetas = rng.uniform(low=-self.rot_range_mag, high=self.rot_range_mag, size=(N,))
            rot_xf = rotation_matrices(thetas)

            if self.scaled_rotation:
                # Original
                xf = cat_nx2x3(xf, rot_xf)
            else:
                # Maybe more correct
                xf = cat_nx2x3(rot_xf, xf)

        return xf.astype(np.float32)

    def aug_colour_xforms(self, N, rng=None):
        if rng is None:
            rng = np.random
        light_offset = np.zeros((N,))
        if self.light_off_std > 0.0:
            light_offset = rng.normal(scale=self.light_off_std, size=(N,))

        light_scale = np.ones((N,))
        if self.light_scl_std > 0.0:
            light_scale = np.exp(rng.normal(loc=0.0, scale=self.log_light_scl_std, size=(N,)))

        return light_scale.astype(np.float32), light_offset.astype(np.float32)

    def aug_colour_xforms_with_pca(self, N, whiten_pca, rng=None):
        if rng is None:
            rng = np.random
        light_offset = np.zeros((N, 3))

        if self.light_off_std > 0.0:
            # Colour offset
            light_offset = rng.normal(scale=0.1, size=(N, 3))
            light_offset = np.dot(light_offset, np.sqrt(whiten_pca.explained_variance_[:, None]) * whiten_pca.components_)

        light_scale = np.ones((N,))
        if self.light_scl_std > 0.0:
            light_scale = np.exp(rng.normal(loc=0.0, scale=self.log_light_scl_std, size=(N,)))

        return light_scale.astype(np.float32), light_offset.astype(np.float32)

