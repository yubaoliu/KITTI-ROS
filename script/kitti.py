#!/usr/bin/env python
# coding=UTF-8
import cv2
import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import rospy
from cv_bridge import CvBridge
import numpy as np
from collections import deque

from data_utils import *
from publish_utils import *
from kitti_util import *
from misc import *

DATA_PATH = '/home/yubao/data/Dataset/KITI/RawData/2011_09_26/2011_09_26_drive_0005_sync'


class Object():
    def __init__(self, center, max_length=20):
        self.max_length = max_length
        # 物体在所有过去的轨迹
        # self.locations = []
        self.locations = deque(maxlen=self.max_length)
        self.locations.appendleft(center)

    def update(self, center, displacement, yaw):
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw_change) + y0 * \
                np.sin(yaw_change) - displacement
            y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
            self.locations[i] = np.array([x1, y1])
        # current frame 当前的观察
        # self.locations += [np.array([0, 0])]
        # only show the last 20 frames
        # self.locations = self.locations[-20:]

        # for car
        # self.locations.appendleft(np.array([0, 0]))

        # For all the objects
        if center is not None:
            self.locations.appendleft(center)

    def reset(self):
        # 清空过去的轨迹
        # self.locations = []
        self.locations = deque(maxlen=self.max_length)


EGOCAR = np.array([[2.15, 0.9, -1.73], [2.15, -0.9, -1.73], [-1.95, -0.9, -1.73], [-1.95, 0.9, -1.73],
                   [2.15, 0.9, -0.23], [2.15, -0.9, -0.23], [-1.95, -0.9, -0.23], [-1.95, 0.9, -0.23]])

if __name__ == '__main__':
    rospy.init_node('kitti_node', anonymous=False)
    bridge = CvBridge()
    rate = rospy.Rate(10)
    frame = 0

    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_pcd', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d', MarkerArray, queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc', MarkerArray, queue_size=10)
    dist_pub = rospy.Publisher('kitti_dist', MarkerArray, queue_size=10)

    df_tracking = read_tracking(
        '/home/yubao/data/catkin_ws/src/kitti_tutorial/data/0000.txt')
    calib = Calibration(
        '/home/yubao/data/Dataset/KITI/RawData/2011_09_26/', from_video=True)

    tracker = {}  # trackid: Object, 记录当前场景中的所有物体
    prev_imu_data = None

    while not rospy.is_shutdown():
        rospy.loginfo("------%d----" % frame)
        df_tracking_frame = df_tracking[df_tracking.frame == frame]

        # read  raw datasets
        image = cv2.imread(os.path.join(
            DATA_PATH, 'image_02/data/%010d.png' % frame))
        point_cloud = read_pcd(os.path.join(
            DATA_PATH, 'velodyne_points/data/%010d.bin' % frame))
        imu_data = read_imu(os.path.join(
            DATA_PATH, 'oxts/data/%010d.txt' % frame))

        # bounding box
        boxes = np.array(
            df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        types = np.array(df_tracking_frame['type'])
        track_ids = np.array(df_tracking_frame['track_id'])

        # 3D boxes
        centers = {}    # track_id: center, 只记录当前帔所侦测到的物体
        corners_3d_velos = []
        # 物体间最小距离, 点P及点Q，以及其距离d
        minPQDs = []

        boxes_3d = np.array(df_tracking_frame[[
            'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        for track_id, box_3d in zip(track_ids, boxes_3d):
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos += [corners_3d_velo]
            # corners_3d_velo: &*3
            centers[track_id] = np.mean(corners_3d_velo, axis=0)[:2]
# distance
            minPQDs += [min_distance_cuboids(EGOCAR, corners_3d_velo)]

        # show the car
        centers[-1] = np.array([0, 0])
        track_ids = np.append(track_ids, -1)
        corners_3d_velos += [EGOCAR]
        types = np.append(types, 'Car')

        if prev_imu_data is None:  # 没有前一帧
            # 初始休第一帧中的物体对象
            for track_id in centers:
                tracker[track_id] = Object(centers[track_id], 50)
        else:
            displacement = 0.1 * np.linalg.norm(imu_data[['vf', 'vl']])
            yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
            # 如果在当前帧中又被观测到
            for track_id in centers:
                if track_id in tracker:  # 如果之前被监测到过
                    tracker[track_id].update(
                        centers[track_id], displacement, yaw_change)
                else:  # 如果是首次被监测到过
                    tracker[track_id] = Object(centers[track_id], 50)

            # 更新过去被监测到，但在当前帧没有出现的物体的坐标
            for track_id in tracker:
                if track_id not in centers:
                    # 不知道其在当前帧中的中心点的位置
                    tracker[track_id].update(None, displacement, yaw_change)

        prev_imu_data = imu_data

        # publish results
        publish_image(cam_pub, bridge, image, boxes, types)
        # Publish point cloud
        publish_pcd(pcl_pub, point_cloud)
        # publish marke
        publish_ego_car(ego_pub)
        # IMU
        publish_imu(imu_pub, imu_data)
        publish_gps(gps_pub, imu_data)
        publish_3dbox(box3d_pub, corners_3d_velos, track_ids, types, True)
        # publish distance
        publish_dist(dist_pub, minPQDs)

        # centers: 当前帧中监测到的物体，tracker所有物体
        publish_loc(loc_pub, tracker, centers)

        rate.sleep()

        frame += 1
        if frame == 154:
            frame = 0
            for track_id in tracker:
                tracker[track_id].reset()
