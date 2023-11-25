import zmq
import time
import cv2
import numpy as np
import json
import random
from scipy.spatial.transform import Rotation as R
from blender_bridge import BlenderBridge

class FrameHandler():
    def __init__(self):
        self.blender_bridge = BlenderBridge()

        ip="127.0.0.1"
        frame_pub_ip="192.168.55.100"
        port=5560
        pub_url = "tcp://{}:{}".format(ip, port)
        sub_url = "tcp://{}:{}".format(ip, port+1)
        frame_pub_url = "tcp://{}:{}".format(ip, port+2)
        raw_fram_pub_url = "tcp://{}:{}".format(ip, port+3)
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.pub_socket.connect(pub_url)
        print("pub connected to: {}\n".format(pub_url))

        self.frame_pub_socket = self.ctx.socket(zmq.PUB)
        self.frame_pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.frame_pub_socket.connect(frame_pub_url)
        print("frame pub connected to: {}\n".format(frame_pub_url))

        self.raw_frame_pub_socket = self.ctx.socket(zmq.PUB)
        self.raw_frame_pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.raw_frame_pub_socket.connect(raw_fram_pub_url)
        print("raw frame pub connected to: {}\n".format(raw_fram_pub_url))

        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.connect(sub_url)
        print("sub connected to: {}\n".format(sub_url))

    def publish_frame(self, image_frame):
        
        image_frame = image_frame.tobytes()
        self.frame_pub_socket.send(image_frame)
    
    def publish_raw_frame(self, image_frame):
        image_frame = cv2.resize(image_frame, (480,360))
        image_frame = image_frame.tobytes()
        self.raw_frame_pub_socket.send(image_frame)

    def T_inv(self, H):
        rot = H[0:3,0:3]
        trans = H[0:3,3].reshape((3,1))
        H_inv = np.hstack((rot.T, -np.matmul(rot.T,trans)))
        H_inv = np.vstack((H_inv,np.array([0,0,0,1])))
        return H_inv

    # def calc_corner_in_W(self, pos_gate,orientation_gate_xyzw):
    #     width, height = 0.5, 0.5
    #     rot = R.from_quat(orientation_gate_xyzw)
    #     T = np.zeros((4,4))
    #     T[0:3,0:3] = rot.as_dcm()
    #     T[0:3,3] = pos_gate.T
    #     T[3,3] = 1
    #     gate_corner_local = np.array([
    #                                 [-width/2, 0, height/2,1],
    #                                 [width/2,0, height/2,1],
    #                                 [width/2,0,-height/2,1],
    #                                 [-width/2,0,-height/2,1]
    #                                 ]).T
    #     gate_corner_world = np.matmul(T,gate_corner_local)
        
    #     return gate_corner_world.T


    def calc_corner_in_G(self, pos_gate,orientation_gate_xyzw):
        width, height = 0.63, 0.63
        rot = R.from_quat(orientation_gate_xyzw)
        T = np.zeros((4,4))
        T[0:3,0:3] = rot.as_dcm()
        T[0:3,3] = pos_gate.T
        T[3,3] = 1
        gate_corner_local = np.array([
                                    [-width/2, 0, height/2,1],
                                    [width/2,0, height/2,1],
                                    [width/2,0,-height/2,1],
                                    [-width/2,0,-height/2,1]
                                    ]).T
        gate_corner_world = np.matmul(T,gate_corner_local)
        
        return gate_corner_local

    def pnp(self, image_points,world_points,camera_matrix):
        # 3D points of the window in the world coordinate system
        # Assuming the window is a rectangle of size width x height

        # # Camera intrinsic parameters
        # camera_matrix = np.array([[50, 0, 480],
        #                         [0, 50, 360],
        #                         [0,  0,  1]])

        width, height = 0.5, 0.5

        gate_corner_local = np.array([
                            [-width/2, 0, height/2],
                            [width/2,0, height/2],
                            [width/2,0,-height/2],
                            [-width/2,0,-height/2]
                            ])

        dist_coeffs = np.zeros((4,))  # assuming no lens distortion

        # Solve for pose
        # ret, rvec, tvec = cv2.solvePnP(world_points, image_points.astype(float), camera_matrix, dist_coeffs)
        ret, rvec, tvec = cv2.solvePnP(gate_corner_local, image_points.astype(float), camera_matrix, dist_coeffs)

        # rot = R.from_rotvec(rvec[0:3,0])
        # T_W2C = np.eye(4)
        # T_W2C[0:3,0:3] = rot.as_dcm()
        # T_W2C[0:3,3] = tvec.T
        # T_C2W = self.T_inv(T_W2C)
        
        # R_camera_in_W = T_C2W[0:3,0:3]
        # pos_camera_in_W = T_C2W[0:3,3]
        # R_cv2blender = np.array([[1, 0, 0], 
        #                         [0, -1, 0], 
        #                         [0, 0, -1]])
        # R_blender = np.matmul(R_camera_in_W, R_cv2blender)
        # rot = R.from_dcm(R_blender)
        # blender_quaternion_xyzw = rot.as_quat()

        rot = R.from_rotvec(rvec[0:3,0])
        T_G2C = np.eye(4)
        T_G2C[0:3,0:3] = rot.as_dcm()
        T_G2C[0:3,3] = tvec.T
        T_C2G = self.T_inv(T_G2C)
        
        R_camera_in_G = T_C2G[0:3,0:3]
        pos_camera_in_G = T_C2G[0:3,3]
        R_cv2blender = np.array([[1, 0, 0], 
                                [0, -1, 0], 
                                [0, 0, -1]])
        R_blender = np.matmul(R_camera_in_G, R_cv2blender)
        rot = R.from_dcm(R_blender)
        blender_quaternion_xyzw = rot.as_quat()

        # print("Position", pos_camera_in_W.tolist())
        # print("Quaternion:", blender_quaternion_xyzw)
        return pos_camera_in_G.tolist(),blender_quaternion_xyzw.tolist()
    
    def handle_frame(self, image_frame, current_gate_idx = 0):
        image_encoded = image_frame.tobytes()
        self.pub_socket.send(image_encoded)
        gates_json = None
        try:
            json_string = self.sub_socket.recv(flags=zmq.NOBLOCK)
            # print(json_string)
            gates_json = json.loads(json_string.decode('utf-8'))
        except zmq.Again as e:
                # print("No Result")
                pass
        
        if gates_json is not None:
            for gate_idx,gate in enumerate(gates_json["gates"]):
                    all_corners_present=True
                    image_corners = []
                    cv2.circle(image_frame,(gate["gate_center"][0][0],gate["gate_center"][0][1]),8,self.gate_color ,6)
                    cv2.rectangle(image_frame, (430,310), (530,410), color=self.gate_color, thickness=2)
                    for corner_idx,corner in enumerate(gate["gate_corners"]):
                        if len(corner)>0:
                            cv2.circle(image_frame,(corner[0][0],corner[0][1]),8,self.colors[corner_idx],6)
                            image_corners.append(corner[0])
                        else:
                            all_corners_present = False

                    if gate_idx == 0:
                         self.closest_gate = gate
                         self.closest_gate_all_corners_present = all_corners_present
                    
                    if all_corners_present == True and gate_idx == 0:
                        camera_pos, camera_rot_xyzw = self.pnp(np.array(image_corners),self.gates_corners_world[current_gate_idx],self.camera_K)
                        self.current_camera_pose = [camera_rot_xyzw[3], camera_rot_xyzw[0], camera_rot_xyzw[1], camera_rot_xyzw[2],camera_pos[0],camera_pos[1],camera_pos[2]]
                        self.camera_states = [[camera_pos[0],camera_pos[1],camera_pos[2], camera_rot_xyzw[3], camera_rot_xyzw[0], camera_rot_xyzw[1], camera_rot_xyzw[2]]]
                        if self.blender_frame == 0:
                            self.blender_bridge.getRender(self.camera_states, self.gate_states,
                                                    render = False,
                                                    render_depth = False, render_seg = False, render_opt = False,
                                                    reset_animation = True,
                                                    change_camera_setting = True,
                                                    image_width = 960,
                                                    image_height = 720,
                                                    K = self.camera_K.tolist())
                        else:
                            self.blender_bridge.getRender(self.camera_states, self.gate_states,
                                                    render = False,
                                                    render_depth = False, render_seg = False, render_opt = False,
                                                    reset_animation = False,
                                                    change_camera_setting = False,
                                                    image_width = 960,
                                                    image_height = 720,
                                                    K = self.camera_K.tolist())
                        self.blender_frame += 1
        else:
            self.closest_gate = None
        # image_frame = cv2.resize(image_frame, (480,360))
        # image_frame = image_frame.tobytes()
        # self.frame_pub_socket.send(image_frame)
        # cv2.imshow("im",image_frame)
        # cv2.waitKey(1)

# vidcap = cv2.VideoCapture('/home/pear/drones/outputs/washburn_drone.mp4')
# success,image = vidcap.read()

# gate_states=[[0.65,1.83,1.36,0.9816,0.,0.,0.1908],
#                             [-0.55, 3.57, 1.36, 1., 0., 0., 0.],
#                             [0.6, 5.59, 1.36, 0.9890, 0.,0.,0.1478]]

# frame_handler = FrameHandler(gate_states)

# while success:
#     frame_handler.handle_frame(image)


#     # gate_detector.detect_gate(image)
#     success,image = vidcap.read()
#     time.sleep(0.03)