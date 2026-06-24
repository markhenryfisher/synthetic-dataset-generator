import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import ops
from torchvision.utils import draw_bounding_boxes
import os

# -------------- Visualization utils ----------------

def show_img(img, name='', cmap='viridis', normalize=True, folder=''):

    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')
    else:        
        pass

    if torch.is_tensor(img):
        img = img.cpu().detach()
        img = img.numpy().squeeze()
    # img = rescale_img(img)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if normalize:            
        plt.imshow(img, cmap=cmap)
    else: 
        plt.imshow(img, cmap=cmap, vmin=0., vmax=1.)          

    
    plt.title('Img: {}'.format(name))

    if folder:    
        filename = name +'.png'
        fullfile = os.path.join(folder, filename)        
        plt.savefig(fullfile)  # Saves as a PNG file
        
    
    return

def display_img(img, name='', cmap='viridis', normalize=True, folder=''):
    show_img(img, name=name, cmap=cmap, normalize=normalize, folder=folder)

    plt.show(block=False)
    plt.pause(0.001)

def plot_hist(data, n_bins=20, rng=None, xlabel='', title=''):
    
    fig, axs = plt.subplots()    
    axs.hist(data, bins=n_bins+1, range=rng, align='left')
    
    plt.xlabel(xlabel)
    plt.ylabel('Fish')
    plt.title(title)
    plt.show()
    
    
            
def show_GT(img, tgt):
    
    print('Ground Truth:')
    density = tgt['density']
    # give each tgt an idx
    idxs = list(range(len(density)))
    labels = ['Box {}'.format(idx) for idx in idxs]        
    for idx in idxs:
        print('Box {}, NMS Threshold: {:.2f}'.format(idx, density[idx]))
 
    gt_bb = draw_bounding_boxes(img.mul(255).to(torch.uint8), tgt['boxes'], labels=labels, width=2)
    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')
    gt_bb = gt_bb.permute(1, 2, 0).byte().numpy()
    plt.imshow(gt_bb)
    # plt.title('nms mode = ' + exp_params['box_nms_mode'])
    plt.title('GT')
    plt.show()
    

def show_bboxes(img, bboxes, name='', fig_folder=''):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if type(img) == np.ndarray:
        img = torch.from_numpy(img).permute(2,0,1)
    
    gt_bb = draw_bounding_boxes(img.mul(255).to(torch.uint8), bboxes, width=2)
    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')
    gt_bb = gt_bb.permute(1, 2, 0).byte().numpy()
    show_img(gt_bb, name)
    
    if fig_folder:
        filename = name +'.png'
        fullfile = os.path.join(fig_folder, filename)        
        plt.savefig(fullfile)  # Saves as a PNG file

    plt.show()
    
    
def show_obbs(img, obbs, name='', fig_folder=''):
    """ 
    show_obbs - visualize oriented bounding boxes
    """
    

    for obb in obbs:
        # add extra vertex to enclose obb
        obb = np.vstack((obb, obb[0]))
        x = obb[:,0]
        y = obb[:,1]
        # mhf 10.11.25 add single point to show bbox orientation
        plt.plot(obb[0,0], obb[0,1], 'ro')
        plt.plot(x, y, color='green', linewidth=2)
        
    show_img(img, name)

    if fig_folder:
        filename = name +'.png'
        fullfile = os.path.join(fig_folder, filename)        
        plt.savefig(fullfile)  # Saves as a PNG file
        
    plt.show()
    
    

def display_matches(proposals, matches, img_shapes):
    """
    mhf 17.11.24 display matches
    
    proposals: List
    matches: List[Tensor] 0: backgraound; 1: foreground
    size: tuple

    Returns
    -------
    None.

    """
    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')

    for s, p, m in zip(img_shapes, proposals, matches):
        y0_sz, x0_sz = s[0], s[1]
        
        # create an image
        img = torch.ones((3, y0_sz, x0_sz), dtype=torch.uint8, device='cpu')
        img = img * 255

        matched_idxs,  = torch.nonzero(m, as_tuple=True)
        bboxes = p[matched_idxs]
        img = draw_bounding_boxes(img, bboxes, width=2)
        
        
        img = img.permute(1, 2, 0).byte().numpy()
        plt.imshow(img)
        
        plt.title('Matches')
        plt.show()
            


def display_per_channel_feat_map(fmap, figsize=(16, 8)):
    """
    display ANY featuremap
    mhf buggy!!!!

    """
    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')
    assert isinstance(fmap, list)
    if not fmap:
        raise RuntimeError('dislay_feat_map: fmap list empty')
    
    # rows == scale; cols == channel
    nrows, ncols = (len(fmap), fmap[0].size(1)  )
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, img in enumerate(axs.flat):
        (row, col) = divmod(i, ncols)
        feat = fmap[row]
        img = feat[:, col, :, :].cpu().detach()
        img = img.permute(1, 2, 0).numpy().squeeze()
        img = rescale_img(img)
        axs[i].imshow(img)
    plt.show()
        

def rescale_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
    
def display_feat_map(fmap, name):
    """
    display feature map

    Parameters
    ----------
    fmap : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # is code running in Spyder?
    if any('SPYDER' in name for name in os.environ):
        # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
        get_ipython().run_line_magic('matplotlib', 'inline')

    
    img = fmap.cpu().detach()
    img = img.numpy().squeeze()
    img = rescale_img(img)
    plt.imshow(img)
    
    plt.title('Fmap Scale {}'.format(name))
    plt.show()



    

def display_imgs(img_list):
    # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    nrows, ncols = (1, len(img_list))
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8))
    for i, img in enumerate(img_list):
        if type(img) == torch.Tensor:
            img = img.cpu().detach().numpy().squeeze()
            if len(img.shape) > 2:
                img = img.permute(1, 2, 0)
        if np.min(img) < 1:
            img = rescale_img(img)
        if ncols == 1: 
            axs.imshow(img*255, cmap='gray', vmin=0, vmax=255)
        else:
            axs[i].imshow()
    plt.show()

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1
        
    return fig, ax


        
    

def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax