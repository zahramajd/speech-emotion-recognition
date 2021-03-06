from PIL import Image
import numpy as np
import os
import pandas as pd

def crop(path):

    img = Image.open(path)
    gray = img.convert("L")

    np_img = np.array(img)
    np_gray = np.array(gray)

    x_min = 0
    y_min = 0
    x_max = np_gray.shape[1] - 1
    y_max = np_gray.shape[0] - 1

    for i in range(np_gray.shape[0]):
        if (np.all(np_gray[i]==255)):
            y_min += 1
        else:
            break

    for i in range(np_gray.shape[1]):
        if (np.all(np_gray[:,i]==255)):
            x_min += 1
        else:
            break

    for i in range(np_gray.shape[0]):
        if (np.all(np_gray[np_gray.shape[0] - i-1]==255)):
            y_max -= 1
        else:
            break

    for i in range(np_gray.shape[1]):
        if (np.all(np_gray[:,np_gray.shape[1]-i-1]==255)):
            x_max -= 1
        else:
            break

    img_crop = np_img[y_min:y_max, x_min:x_max]
    cropped = Image.fromarray(img_crop)

    return cropped

def crop_all():

    for filename in os.listdir('spectrograms-torento'):
        if(not filename =='.DS_Store'):
            cropped = crop('spectrograms-torento/'+filename)
            cropped.save('prep-spectrograms-torento/'+filename)

    return

# crop_all()
all_data = np.empty((0,2))
for filename in os.listdir('prep-spectrograms-torento'):
    if(not filename =='.DS_Store'):
        label = filename.split('.')[0].split('_')[2]
        all_data = np.append(all_data, [[filename, label]], axis = 0)

pd.DataFrame(all_data).to_csv('all_data_torento.csv', header=False, index=False)