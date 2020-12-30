#!/usr/bin/env python
# coding=UTF-8
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
import tf
import cv2

PCD_FRAME_ID = 'map'

COLOR_DICT = {'Car': (255, 255, 0), 'Cyclist': (
    255, 0, 255), 'Pedestrian': (255, 55, 25)}

# because ros rate is set to 10
LIFETIME = 0.1

LINES = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
    # Connections between upper and lower planes
        [0, 4], [1, 5], [2, 6], [3, 7]
]


def publish_image(image_pub, bridge, image, boxes, types):
    for typ, box in zip(types, boxes):
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        cv2.rectangle(image, top_left, bottom_right, COLOR_DICT[typ], 2)

    image_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))


def publish_pcd(pcl_pub, point_cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = PCD_FRAME_ID
    # 排除反射度, 第一个':'表示要取全部的点，第二个‘：’表示取前三列, 最终结果是个n*3的矩阵
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))


def publish_ego_car(ego_car_pub):
    marker_array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = PCD_FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2

    # 参考velodyne 的坐标系
    marker.points = []
    marker.points.append(Point(10, -10, 0))
    marker.points.append(Point(0, 0, 0))
    marker.points.append(Point(10, 10, 0))

    marker_array.markers.append(marker)

    # add car model
    mesh_marker = Marker()
    mesh_marker.header.frame_id = PCD_FRAME_ID
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "package://kitti_tutorial/3dModels/AC_Cobra_269/ShelbyWD.dae"
    # mesh_marker.mesh_resource = "package://kitti_tutorial/Audi_R8/Models/Audi R8.dae"

    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73

    q = tf.transformations.quaternion_from_euler(0, 0, np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 0.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 1.0
    mesh_marker.scale.y = 1.0
    mesh_marker.scale.z = 1.0

    marker_array.markers.append(mesh_marker)

    # publish marker
    ego_car_pub.publish(marker_array)


def publish_imu(imu_pub, imu_data):
    imu = Imu()
    imu.header.frame_id = PCD_FRAME_ID
    imu.header.stamp = rospy.Time.now()

    q = tf.transformations.quaternion_from_euler(
        float(imu_data.roll), float(imu_data.pitch), float(imu_data.yaw))
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)


def publish_gps(gps_pub, imu_data):
    gps = NavSatFix()

    gps.header.frame_id = PCD_FRAME_ID
    gps.header.stamp = rospy.Time.now()

    gps.latitude = imu_data.lat
    gps.longitude = imu_data.lon
    gps.altitude = imu_data.alt

    gps_pub.publish(gps)


def publish_3dbox(box3d_pub, corners_3d_velos, track_ids, types=None, publish_id=True):
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = PCD_FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        b, g, r = COLOR_DICT[types[i]]
        if types is None:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        else:
            marker.color.r = r/255.0
            marker.color.g = g/255.0
            marker.color.b = b/255.0
        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

        if publish_id:
            # track id
            text_marker = Marker()
            text_marker.header.frame_id = PCD_FRAME_ID
            text_marker.header.stamp = rospy.Time.now()

            text_marker.id = i + 1000
            text_marker.action = Marker.ADD
            text_marker.lifetime = rospy.Duration(LIFETIME)
            text_marker.type = Marker.TEXT_VIEW_FACING

            # upder front left corner
            # p4 = corners_3d_velo[4]
            # average
            p = np.mean(corners_3d_velo, axis=0)

            text_marker.pose.position.x = p[0]
            text_marker.pose.position.y = p[1]
            text_marker.pose.position.z = p[2] + 1

            text_marker.text = str(track_ids[i])

            text_marker.scale.x = 1
            text_marker.scale.y = 1
            text_marker.scale.z = 1

            b, g, r = COLOR_DICT[types[i]]
            text_marker.color.r = r/255.0
            text_marker.color.g = g/255.0
            text_marker.color.b = b/255.0
            text_marker.color.a = 1.0
            text_marker.scale.x = 0.1
            marker_array.markers.append(text_marker)

    box3d_pub.publish(marker_array)


def publish_loc(loc_pub, tracker, centers):
    marker_array = MarkerArray()

    for track_id in centers:
        marker = Marker()
        marker.header.frame_id = PCD_FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration()
        marker.type = Marker.LINE_STRIP
        marker.id = track_id

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.2

        marker.points = []
        for p in tracker[track_id].locations:
            marker.points.append(Point(p[0], p[1], 0))
        marker_array.markers.append(marker)

    loc_pub.publish(marker_array)


def publish_dist(dist_pub, minPQDs):
    marker_array = MarkerArray()

    for i, (minP, minQ, minD) in enumerate(minPQDs):
        marker = Marker()
        marker.header.frame_id = PCD_FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_STRIP
        marker.id = i

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.b = 0.5
        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker.points = []
        marker.points.append(Point(minP[0], minP[1], 0))
        marker.points.append(Point(minQ[0], minQ[1], 0))

        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = PCD_FRAME_ID
        text_marker.header.stamp = rospy.Time.now()

        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(LIFETIME)
        text_marker.type = Marker.TEXT_VIEW_FACING

        p = (minP + minQ) / 2.0
        text_marker.pose.position.x = p[0]
        text_marker.pose.position.y = p[1]
        text_marker.pose.position.z = 0.0

        text_marker.text = '%.2f' % minD

        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1

        text_marker.color.r = 0.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        marker_array.markers.append(text_marker)

    dist_pub.publish(marker_array)
