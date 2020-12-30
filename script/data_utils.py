#!/usr/bin/env python
# coding=UTF-8

import cv2
import numpy as np
import pandas as pd

IMU_COLUMN_NAMES = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 'wx', 'wy', 'wz','wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']


TRACKING_COLUMN_NAMES = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                         'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

def read_image(image_path):
    return cv2.imread(image_path)


# 读进来的是一个一维的阵列，需要转成n*4的矩阵, n为点的个数, 每个点有四个信息(x, y, z, 反射度)
def read_pcd(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

def read_imu(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = IMU_COLUMN_NAMES
    return df

def read_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = TRACKING_COLUMN_NAMES

    df = df[df['track_id']>=0] # remove DontCare objects
    df.loc[df.type.isin(['Bus', 'Truck', 'Van', 'Tram']), 'type'] = 'Car' # Set all vehicle type to Car
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    return df
