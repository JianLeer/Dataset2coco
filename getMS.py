# get means/std
import cv2, os, argparse
import numpy as np
from tqdm import tqdm
import glob

def main():
    dirs1 = r'/home/idip/lijian/pytorch/Dataset/VisDrone2019-DET/train/images/'
    dirs2 = r'/home/idip/lijian/pytorch/Dataset/VisDrone2019-DET/val/images/' 
    dirs3 = r'/home/idip/lijian/pytorch/Dataset/VisDrone2019-DET/test/images/' 
    img_paths = glob.glob(os.path.join(dirs1, '*.jpg'))
    img_paths.extend(glob.glob(os.path.join(dirs2, '*.jpg')))
    img_paths.extend(glob.glob(os.path.join(dirs3, '*.jpg')))
    m_list, s_list = [], []
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        # img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean = ", m[0][::-1])
    print("std = ", s[0][::-1])

if __name__ == '__main__':
    main()
