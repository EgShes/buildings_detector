import numpy as np


def crop_windows(x_data, y_data, window_size: tuple, n_channels: int, threshold: float):
    """
    Crops from bigger (N,N,ch) images fewer (n,n,ch) images. It is for x_data
    Crops from bigger (N,N,1) images fewer (n,n,1) images. It is for y_data
    :param x_data: origin images
    :param y_data: masks of origin images
    :param window_size: size of new fewer images
    :param n_channels: n channels of image
    :param threshold: images with less percent of building pixels are not included into new list of cropped images
    :return: (l, n, n, ch) Tensor for images and (l, n, n, 1) Tensor for masks
    """
    HEIGHT, WIDTH = window_size[0], window_size[1]

    x_list = []
    y_list = []

    for i in range(x_data.shape[0]):
        img = x_data[i, :, :, :]
        mask = y_data[i, :, :, :]

        for row in range(3):
            for col in range(3):
                temp_img = np.zeros(shape=(HEIGHT, WIDTH, n_channels))
                temp_mask = np.zeros(shape=(HEIGHT, WIDTH, 1))
                y0, y1 = row * HEIGHT, (row+1) * HEIGHT
                x0, x1 = col * WIDTH, (col+1) * WIDTH
                (avlbl_y, avlbl_x, _) = img[y0:y1, x0:x1, :].shape
                temp_mask[:avlbl_y, :avlbl_x, :] = mask[y0:y1, x0:x1, :]

                if len(temp_mask[temp_mask == 1]) / len(temp_mask[temp_mask == 0]) < threshold:
                    continue

                temp_img[:avlbl_y, :avlbl_x, :] = img[y0:y1, x0:x1, :]
                x_list.append(np.copy(temp_img))
                y_list.append(np.copy(temp_mask))

    length = len(x_list)
    x_data_proc = np.empty(shape=(length, HEIGHT, WIDTH, n_channels))
    y_data_proc = np.empty(shape=(length, HEIGHT, WIDTH, 1))

    for i in range(length):
        x_data_proc[i, :, :, :] = x_list[i]
        y_data_proc[i, :, :, :] = y_list[i]

    return x_data_proc, y_data_proc


def crop_windows_2(x_data, y_data, window_size: tuple, n_channels: int,
                   threshold: float, percent_of_empty_images: float):
    """
    Crops from bigger (N,N,ch) images fewer (n,n,ch) images. It is for x_data
    Crops from bigger (N,N,1) images fewer (n,n,1) images. It is for y_data
    Controls how much no building examples will be in return tensors
    :param x_data: origin images
    :param y_data: masks of origin images
    :param window_size: size of new fewer images
    :param n_channels: n channels of image
    :param threshold: images with less percent of building pixels are not included into new list of cropped images
    :param percent_of_empty_images: percent of no building images
    :return: (l, n, n, ch) Tensor for images and (l, n, n, 1) Tensor for masks
    """
    from random import shuffle

    HEIGHT, WIDTH = window_size[0], window_size[1]

    x_list_full = []
    y_list_full = []
    x_list_empty = []
    y_list_empty = []

    for i in range(x_data.shape[0]):
        img = x_data[i, :, :, :]
        mask = y_data[i, :, :, :]

        for row in range(3):
            for col in range(3):
                temp_img = np.zeros(shape=(HEIGHT, WIDTH, n_channels))
                temp_mask = np.zeros(shape=(HEIGHT, WIDTH, 1))
                y0, y1 = row * HEIGHT, (row+1) * HEIGHT
                x0, x1 = col * WIDTH, (col+1) * WIDTH
                (avlbl_y, avlbl_x, _) = img[y0:y1, x0:x1, :].shape
                temp_mask[:avlbl_y, :avlbl_x, :] = mask[y0:y1, x0:x1, :]

                if len(temp_mask[temp_mask == 1]) / len(temp_mask[temp_mask == 0]) < threshold:
                    temp_img[:avlbl_y, :avlbl_x, :] = img[y0:y1, x0:x1, :]
                    if len(temp_img[temp_img == 0]) / len(temp_img) < 0.2:
                        x_list_empty.append(np.copy(temp_img))
                        y_list_empty.append(np.copy(temp_mask))
                    continue

                temp_img[:avlbl_y, :avlbl_x, :] = img[y0:y1, x0:x1, :]
                x_list_full.append(np.copy(temp_img))
                y_list_full.append(np.copy(temp_mask))

    if int(percent_of_empty_images * len(x_list_full)) > len(x_list_empty):
        a = len(x_list_empty)
    else:
        a = int(percent_of_empty_images * len(x_list_full))

    length = len(x_list_full) + a
    x_data_proc = np.empty(shape=(length, HEIGHT, WIDTH, n_channels))
    y_data_proc = np.empty(shape=(length, HEIGHT, WIDTH, 1))

    full_indxs = list(range(len(x_list_full)))
    empty_indxs = list(range(len(x_list_empty)))
    shuffle(full_indxs)
    shuffle(empty_indxs)
    l = len(x_list_full)
    for i in range(length):
        if i < l:
            x_data_proc[i, :, :, :] = x_list_full[i]
            y_data_proc[i, :, :, :] = y_list_full[i]
        else:
            x_data_proc[i, :, :, :] = x_list_empty[i - l]
            y_data_proc[i, :, :, :] = y_list_empty[i - l]

    import sklearn.utils
    x_data_proc, y_data_proc = sklearn.utils.shuffle(x_data_proc, y_data_proc, random_state=42)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(20, 2, figsize=(5, 20))
    for i in range(20):
        ax[i][0].imshow(x_data_proc[i, :, :, 0])
        ax[i][0].axis('off')
        ax[i][1].imshow(y_data_proc[i, :, :, 0])
        ax[i][1].axis('off')
    plt.show()

    return x_data_proc, y_data_proc


def assemble_test(x):
    """
    From 9 test images (n,n,ch) or test masks (n,n,1) assembles 1 image (N,N,ch) or mask (N,N,1)
    :param x: One tensor (x_test or y_test or y_predicted)
    :return: Tensor (l, N, N, ch) or Tensor(l, N, N, 1)
    """
    new_lenght = x.shape[0]
    shape = x.shape[1:]

    ret = np.empty(shape=(new_lenght, 650, 650, shape[2]))

    for i in range(new_lenght // 9):
        img = np.zeros(shape=(3 * shape[0], 3 * shape[1], shape[2]))
        for row in range(3):
            for col in range(3):
                y0 = row * shape[0]
                y1 = (row + 1) * shape[0]
                x0 = col * shape[1]
                x1 = (col + 1) * shape[1]
                cur_n = 9*i + 3*row + col
                if shape[2] == 1:
                    img[y0:y1, x0:x1, 0] = x[cur_n, :, :, 0]
                else:
                    img[y0:y1, x0:x1, :] = x[cur_n, :, :, :]
        if shape[2] == 1:
            ret[i, :, :, 0] = img[:650, :650, 0]
        else:
            ret[i, :, :, :] = img[:650, :650, :]
    return ret

