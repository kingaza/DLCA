
import os 
import argparse
import zipfile as zf

import numpy as np 
from scipy.ndimage.measurements import center_of_mass, label, find_objects
from scipy.ndimage import rotate, zoom, grey_closing
from skimage import io, measure

import SimpleITK as sitk
import nibabel as nib


# x y z -> z, y, z
def read_image(path):
    # Note: scale image intensity
    image = nib.load(path).get_fdata() / 100.0
    spacing = sitk.ReadImage(path).GetSpacing()
    return image.transpose(2, 1, 0), spacing[::-1]

# x y z -> z, y, z
def read_label(path):
    label = nib.load(path).get_fdata().astype(np.int8).clip(0, 1)
    return label.transpose(2, 1, 0)


def rescale(image, label, input_space, output_space = (0.39, 0.39, 0.39)):
    assert image.shape == label.shape, "image shape:{} != label shape{}".format(image.shape, label.shape)
    zoom_factor = tuple([input_space[i] / output_space[i] for i in range(3)])
    # image cubic interpolation
    image_rescale = zoom(image, zoom_factor, order=3)
    # label nearest interpolation
    label_rescale = zoom(label, zoom_factor, order=0)
    label_rescale = grey_closing(label_rescale, size=(5,5,5))

    return image_rescale, label_rescale



def find_shift(shape, shape_s, center, top):
    to_min_z  = max(shape[0] - top, 0)
    from_min_z = max(top - shape[0], 0)
    to_max_z = shape[0]
    from_max_z = top
    
    to_min_y = max(int(shape[1] / 2) - center[1], 0)
    from_min_y = max(center[1] - int(shape[1] / 2), 0)
    to_max_y = min(shape_s[1] - center[1], int(shape[1] / 2) - 1) + int(shape[1]/2)
    from_max_y = min(shape_s[1] - center[1], int(shape[1] / 2) - 1) + center[1]
    
    to_min_x = max(int(shape[2] / 2) - center[2], 0)
    from_min_x = max(center[2] - int(shape[2] / 2), 0)
    to_max_x = min(shape_s[2] - center[2], int(shape[2] / 2) - 1) + int(shape[2] / 2)
    from_max_x = min(shape_s[2] - center[2], int(shape[2] / 2) - 1) + center[2]

    coord_s = [from_min_z, from_max_z, from_min_y, from_max_y, from_min_x, from_max_x]
    coord_t = [to_min_z, to_max_z, to_min_y, to_max_y, to_min_x, to_max_x]

    coord_s = np.array(coord_s).astype(np.int16)
    coord_t = np.array(coord_t).astype(np.int16)

    return coord_s, coord_t


def crop(image, label, shape=(512, 512, 512)):
    np.clip(label, 0, 1, out = label)
    if shape is None:
        return image, label

    mask = image > 0
    center = tuple(map(int, center_of_mass(mask)))
    max_region = find_objects(mask)[0]
    top_slice = max_region[0].stop
    source_shape = image.shape

    coord_s, coord_t = find_shift(shape, source_shape, center, top_slice)

    # TODO
    image_crop = np.ones(shape, dtype = np.int16) * image.min()
    label_crop = np.zeros(shape, dtype = np.uint8)
    
    image_crop[coord_t[0]:coord_t[1], coord_t[2]:coord_t[3], coord_t[4]:coord_t[5]] = \
        image[coord_s[0]:coord_s[1], coord_s[2]:coord_s[3], coord_s[4]:coord_s[5]]
    
    label_crop[coord_t[0]:coord_t[1], coord_t[2]:coord_t[3], coord_t[4]:coord_t[5]]= \
        label[coord_s[0]:coord_s[1], coord_s[2]:coord_s[3], coord_s[4]:coord_s[5]]

    return image_crop, label_crop



def rotate_center(image, label, angle=15, ax=0):

    assert image.shape == label.shape

    if angle < 1:
        return image, label

    axes = tuple({0,1,2}.difference({ax}))
    # image cubic interpolation
    img = rotate(image, angle, axes=axes, reshape=False, order=3)
    # label nearest interpolation
    lbl = rotate(label, angle, axes=axes, reshape=False, order=0)
    lbl = grey_closing(lbl, size=(5,5,5))

    return img, lbl
  


def write_image(image, path):
    # SimpleITK save data in the format Z-H-W
    image_array = sitk.GetImageFromArray(image.transpose(2, 1, 0))
    sitk.WriteImage(image_array, path)


def draw_bbox(image, boxes, data_name, path):
    for i, box in enumerate(boxes):
        z, h, w, r = int(box[0]), int(box[1]), int(box[2]), int(box[3]/2)
        img_slice = image[z,:,:]
        img_slice = (img_slice - img_slice.min()) * 255 / (img_slice.max()-img_slice.min())
        img_slice = img_slice.astype(np.uint8)
        img_slice = np.repeat(img_slice[..., None], 3, axis=-1)

        shp = img_slice.shape
        if (h-r)>0 and (h+r)<shp[0] and (w-r)>0 and (w+r)<shp[1]:
            line = np.repeat([[255, 0, 0]],2*r,axis=0)
            img_slice[h-r:h+r,w-r,:] = line
            img_slice[h-r:h+r,w+r,:] = line
            img_slice[h-r,w-r:w+r,:] = line
            img_slice[h+r,w-r:w+r,:] = line
            save_path = os.path.join(path, '{}_bbox{}.png'.format(data_name, i+1))
            io.imsave(save_path, img_slice)


def gen_bbox(seg_label):
    label_region, label_num = label(seg_label)
    object_regions = find_objects(label_region)
    boxes = []
    for object_region in object_regions:
        box = []
        max_length = 0
        for i in range(3):
            min_coord = object_region[i].start 
            max_coord = object_region[i].stop 
            center = int(0.5 * (min_coord + max_coord))
            box.append(center)
            length = max_coord - min_coord
            if length > max_length:
                max_length = length
            if i == 2:
                box.append(max_length)
        boxes.append(box)

    return boxes 


def save_result(image, label, image_name, save_root):
    boxes = gen_bbox(label)
    write_image(image, os.path.join(save_root, "{}.nii.gz".format(image_name)))
    write_image(label, os.path.join(save_root, "{}_label.nii.gz".format(image_name)))
    draw_bbox(image, boxes, image_name,save_root)
    np.save(os.path.join(save_root, "{}_bbox.npy".format(image_name)), boxes)
    np.savetxt(os.path.join(save_root, "{}_bbox.txt".format(image_name)), boxes, delimiter=',', fmt='%d')


def preprocess(rotate_type, zoom_type, in_dir, data_file, out_dir):

    if data_file.split('.')[-1] != 'zip':
        return
    
    data_path = os.path.join(in_dir, data_file)

    z = zf.ZipFile(data_path)
    for name in z.namelist():
        if 'pre/TOF.nii.gz' in name:
            z.extract(name, out_dir) 
        if 'aneurysms.nii.gz' in name:
            z.extract(name, out_dir)      
        if 'location.txt' in name:
            z.extract(name, out_dir)      
    z.close()    

    data_name = data_file.split('.')[0]
    tof_path = os.path.join(out_dir, data_name, 'pre/TOF.nii.gz')
    label_path = os.path.join(out_dir, data_name, 'aneurysms.nii.gz')

    image, spacing = read_image(tof_path)
    label = read_label(label_path)

    output_spacing = (0.39, 0.39, 0.39)
    if zoom_type == 1:
        output_spacing = tuple(np.asarray(output_spacing) * 1.11)
    elif zoom_type == 2:
        output_spacing = tuple(np.asarray(output_spacing) * 0.9)
    image_rescale, label_rescale = rescale(image, label, input_space=spacing, output_space=output_spacing)

    ax = (rotate_type+1) // 2
    angle = ((rotate_type+1) % 2 * 2 - 1) * 15
    image_rotate, label_rotate = rotate_center(image_rescale, label_rescale, angle=angle, ax=ax)

    image_crop, label_crop = crop(image_rotate, label_rotate, shape=None)
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(data_file, spacing)
    print(image.shape, '->', image_rescale.shape, '->', image_rotate.shape, '->', image_crop.shape)

    # save
    save_result(image_crop, label_crop, data_name, out_dir)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ca detection')
    parser.add_argument('--input', default='', type=str, metavar='SAVE',
                        help='directory to ADAM 2020 data (default: none)')
    parser.add_argument('--output', default='', type=str, metavar='SAVE',
                        help='directory to save nii.gz data (default: none)')
    parser.add_argument('--rotate-type', default='0', type=int, metavar='S',
                        help='0:None, 1:ax1/15, 2: ax1/-15, 3:ax2/15, 4:ax2/-15')
    parser.add_argument('--zoom-type', default='0', type=int, metavar='S',
                        help='0:None, 1:ZoomOut/10%, 2:ZoomIn/10%')                                                
    
    global args
    args = parser.parse_args()

    rotate_type = args.rotate_type
    zoom_type = args.zoom_type
    input_dir = args.input
    output_dir = args.output + '_{}{}'.format(rotate_type, zoom_type)

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    data_list = os.listdir(input_dir)

    for data_file in data_list:
        preprocess(rotate_type, zoom_type, input_dir, data_file, output_dir)