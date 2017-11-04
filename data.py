import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 

def main():
    x = sio.loadmat('/Users/jasoncjs/Documents/GitHub/unet/chi_phantom1.mat')
    x = x['chi']

    # x = ndi.zoom(x, 0.25)

    for i in range(20):
        if i == 0:
            new = np.concatenate((x[None,:,:,:], datagen(x)[None,:,:,:]))
        else:
            new = np.concatenate((new, datagen(x)[None,:,:,:]))

    sio.savemat('temp.mat', {'temp':new})


def datagen(x):
    w, h, d = x.shape[0], x.shape[1], x.shape[2]
    shift_factor = [np.random.uniform(w/4), np.random.uniform(h/4), np.random.uniform(d/4)]
    x = ndi.shift(x, shift_factor)

    theta = np.random.uniform(45)
    x = ndi.rotate(x, theta, (0,1), reshape=False)

    zoom_factor = np.random.uniform(0.1, 1.9)
    x = clipped_zoom(x, zoom_factor)

    scale_factor = np.random.uniform(0.5, 1.5)
    x = x * scale_factor
    return x


def clipped_zoom(img, zoom_factor):

    h, w, d = img.shape[:3]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))
    zd = int(np.round(zoom_factor * d))

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        front = (d - zd) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw, front:front+zd] = ndi.zoom(img, zoom_factor)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        front = (zd - d) // 2
        out = ndi.zoom(img[top:top+zh, left:left+zw, front:front+zd], zoom_factor)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        trim_front = ((out.shape[2] - d) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w, trim_front:trim_front+d]

    # if zoom_factor == 1, just return the input array
    else:
        out = img

    return out

if __name__ == "__main__": main()