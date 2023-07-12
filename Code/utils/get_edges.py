# Derive GT of boundary (thin - 2*2)

import os
from PIL import Image
import numpy as np

# area_GT_path = '/home/stu/zy/PraNet-master/data/TrainDataset/mask/' # The directory where mask is located
# # /home/stu/zy/PraNet-master/data/TrainDataset/mask
# save_path = './save_dir/thin-2*2/' # Save directory for generated edge images
#
# if not os.path.exists(save_path):
#     os.makedirs((save_path))
#
# for filename in os.listdir(area_GT_path):
#     area_GT = np.array(Image.open(area_GT_path + filename))
#     # [rows, cols] = area_GT.shape # (530，614,3)
#     rows = area_GT.shape[0]
#     cols = area_GT.shape[1]
#     boundary_GT = np.zeros((rows, cols))
#
#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             # flag = (area_GT[i, j] != area_GT[i - 1, j] or area_GT[i, j] != area_GT[i + 1, j] or area_GT[i, j] != area_GT[i, j - 1]or area_GT[i, j] != area_GT[i, j + 1])
#
#             if (area_GT[i, j] != area_GT[i - 1, j]).all() or (area_GT[i, j] != area_GT[i + 1, j]).all() or (area_GT[i, j] != area_GT[i, j - 1]).all() or  (area_GT[i, j] != area_GT[i, j + 1]).all():
#                 boundary_GT[i][j] = 255
#
#     boundary_GT = Image.fromarray(boundary_GT)
#     boundary_GT = boundary_GT.convert("L")
#     boundary_GT.save(os.path.join(save_path, os.path.basename(filename)))


# ################################################################################################################
# # Derive GT of boundary (medium - 3*3)
# import os
# from PIL import Image
# import numpy as np
#
# area_GT_path = '../polyp/images/'
# save_path = './save_dir/medium-3*3/'
#
# if not os.path.exists(save_path):
#     os.makedirs((save_path))
#
# for filename in os.listdir(area_GT_path):
#     area_GT = np.array(Image.open(area_GT_path + filename))
#     rows = area_GT.shape[0]
#     cols = area_GT.shape[1]
#     boundary_GT = np.zeros((rows, cols))
#
#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             if (area_GT[i, j] != area_GT[i - 1, j]).all() or (area_GT[i, j] != area_GT[i + 1, j]).all() or (area_GT[i, j] != area_GT[i, j - 1]).all() or (area_GT[i, j] != area_GT[i, j + 1]).all() \
#                     or (area_GT[i, j] != area_GT[i-1, j-1]).all() or (area_GT[i, j] != area_GT[i-1, j+1]).all() or (area_GT[i, j] != area_GT[i+1, j-1]).all() or (area_GT[i, j] != area_GT[i+1, j+1]).all():
#                 boundary_GT[i][j] = 255
#
#     boundary_GT = Image.fromarray(boundary_GT)
#     boundary_GT = boundary_GT.convert("L")
#     boundary_GT.save(os.path.join(save_path, os.path.basename(filename)))
#
#
# # ################################################################################################################
# Derive GT of boundary (medium - 5*5)

import os
from PIL import Image
import numpy as np

area_GT_path = '/home/stu/zy/data/polyp/TestDataset/Kvasir/masks/'
save_path = '/home/stu/zy/data/polyp/TestDataset/Kvasir/edge_5*5/'

if not os.path.exists(save_path):
    os.makedirs((save_path))

for filename in os.listdir(area_GT_path):
    area_GT = np.array(Image.open(area_GT_path + filename))
    rows = area_GT.shape[0]
    cols = area_GT.shape[1]
    boundary_GT = np.zeros((rows, cols))

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            if (area_GT[i, j] != area_GT[i - 1, j]).all() or (area_GT[i, j] != area_GT[i + 1, j]).all() or (area_GT[i, j] != area_GT[i, j - 1]).all() or (area_GT[i, j] != area_GT[i, j + 1]).all() \
                    or (area_GT[i, j] != area_GT[i-1, j-1]).all() or (area_GT[i, j] != area_GT[i-1, j+1]).all() or (area_GT[i, j] != area_GT[i+1, j-1]).all() or (area_GT[i, j] != area_GT[i+1, j+1]).all() \
                    or (area_GT[i, j] != area_GT[i-2, j-2]).all() or (area_GT[i, j] != area_GT[i-2, j-1]).all() or (area_GT[i, j] != area_GT[i-2, j]).all() or (area_GT[i, j] != area_GT[i-2, j+1]).all() or (area_GT[i, j] != area_GT[i-2, j+2]).all() \
                    or (area_GT[i, j] != area_GT[i-1, j-2]).all() or (area_GT[i, j] != area_GT[i-1, j+2]).all() or (area_GT[i, j] != area_GT[i, j-2]).all() or (area_GT[i, j] != area_GT[i, j+2]).all() \
                    or (area_GT[i, j] != area_GT[i+1, j-2]).all() or (area_GT[i, j] != area_GT[i+1, j+2]).all()or (area_GT[i, j] != area_GT[i+2, j-2]).all() or (area_GT[i, j] != area_GT[i+2, j-1]).all() \
                    or (area_GT[i, j] != area_GT[i+2, j]).all() or (area_GT[i, j] != area_GT[i+2, j+1]).all()or (area_GT[i, j] != area_GT[i+2, j+2]).all():

                boundary_GT[i][j] = 255

    boundary_GT = Image.fromarray(boundary_GT)
    boundary_GT = boundary_GT.convert("L")
    boundary_GT.save(os.path.join(save_path, os.path.basename(filename)))


