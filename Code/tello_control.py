from djitellopy import Tello
import time
from threading import Thread, Lock
from dual_quaternions import DualQuaternion
from of import OpticalFlow
import cv2
from frame_handler import FrameHandler

class FlyTello():
    def __init__(self):
        self.pose = None  # Placeholder for the drone's pose
        # self.gate_count = 0  # Counter for the gates
        self.target_hover_height = 105  # Target hover height after takeoff
        self.height = 0
        self.connection_attempt = 0
        self.take_off_attempts = 0
        self.tello = Tello(retry_count=1)
        self.most_recent_frame = None
        self.of_frame_1 = None 
        self.of_frame_2 = None
        self.streaming = False
        self.stream_thread = None
        self.of_ins = OpticalFlow()
        # self.gate_states=[[0.65,1.83,1.36,0.9816,0.,0.,0.1908],
        #                     [-0.55, 3.57, 1.36, 1., 0., 0., 0.],
        #                     [0.6, 5.59, 1.36, 0.9890, 0.,0.,0.1478]]
        self.frame_lock = Lock()
        self.frame_handler = FrameHandler()
        self.current_camera_pose = None
        # self.Q_W2Drone_current = DualQuaternion.from_quat_pose_array([1., 0., 0., 0., 0., 0., 1.6])

        # Initialize the state machine with the custom state machine class
        # self.machine = Machine(model=self, states=DroneStateMachine.states, transitions=DroneStateMachine.transitions, initial='Disconnected')

    def abort_mission(self):
        print("abort_mission")
        self.stop_streaming()
        self.tello.get_distance_tof()
        for i in range(3):
            self.tello.land()
        self.tello.end()
    
    # def process_frame(self):
    #     self.most_recent_frame = self.tello.get_frame_read().frame
    #     if self.most_recent_frame is not None and self.most_recent_frame.shape[0] == 720:
    #         # print(self.most_recent_frame.shape)
    #         self.frame_handler.handle_frame(self.most_recent_frame, self.gate_count)

    def stream_frame(self):
        self.streaming = True
        # time.sleep(5.0)
        while self.streaming:
            with self.frame_lock:
                try:
                    frame = self.tello.get_frame_read().frame
                except Exception as e:
                    print(e)
                    pass
            
            if frame is not None and frame.shape[0] == 720:
                self.most_recent_frame = frame
                self.frame_handler.publish_raw_frame(self.most_recent_frame)
            else:
                pass
            time.sleep(1/30)
            #     self.frame_handler.publish_frame(self.most_recent_frame)
                # print(self.most_recent_frame.shape)
                # self.frame_handler.handle_frame(self.most_recent_frame, self.gate_count)
            
    
    def stop_streaming(self):
        self.streaming = False
        # cv2.destroyAllWindows()
        self.stream_thread.join()

    def connect_tello(self):
        print("connect_tello")
        try:
            # ENTER COMMAND MODE AND TRY CONNECTING OVER UDP
            self.connection_attempt += 1
            print("Connect attempt ", self.connection_attempt)
            self.tello.connect()
            self.tello.set_video_fps(Tello.FPS_30)
            self.tello.streamon()
            print('Battery, ', self.tello.get_battery())
        except:
            print('Failed to connect or it connected but "ok" not received. Retrying...')
            if self.connection_attempt > 3:
                print('Failed to connect after multiple attempts')
                exit(-1)
            return False
        else:
            self.stream_thread = Thread(target=self.stream_frame)
            self.stream_thread.start()
            # time.sleep(5.0)
            print("Connection successful")
            return True

    def check_take_off_success(self):
        self.height = self.tello.get_distance_tof()
        time.sleep(0.05)
        return self.height > 70.0
    
    def attempt_take_off(self):
        print("attempt_take_off")
        take_off_success = False
        while not take_off_success and self.take_off_attempts < 3:
            try:
                self.tello.takeoff()
            except Exception as e:
                print(e)
                pass
            time.sleep(0.05)
            self.tello.send_rc_control(0,0,0,0)
            self.take_off_attempts += 1
            start_time = time.time()
            while True:
                el_time = time.time() - start_time
                if self.check_take_off_success():
                    # time.sleep(7)
                    current_altitude = self.tello.get_distance_tof()
                    try:
                        self.tello.move_up(self.target_hover_height-current_altitude)
                    except Exception as e:
                        print(e)
                        pass
                    print('Takeoff complete in seconds = ', el_time)
                    print('Altitude ', self.tello.get_distance_tof())
                    take_off_success = True
                    self.Q_W2Drone_current = DualQuaternion.from_quat_pose_array([1., 0., 0., 0., 0., 0., 1.6])
                    break
                elif el_time > 8.0:
                    break
                else:
                    # sleep for 1 second and check again
                    time.sleep(0.05)
    
    def move_to_rendezvous(self):
        print("move_to_rendezvous")
        # try:
        #     self.tello.rotate_counter_clockwise(90)
        # except Exception as e:
        #     print(e)
        #     pass
        
        try:
            self.tello.move_forward(70)
        except Exception as e:
            print(e)
            pass
        time.sleep(0.5)
        self.tello.send_rc_control(0,0,0,0)
    
    def move_forward(self,distance):
        try:
            self.tello.move_forward(distance)
        except Exception as e:
            print(e)
            pass
        time.sleep(0.5)
        self.tello.send_rc_control(0,0,0,0)
        
    def move_left(self,distance):
        print("Moving left")
        try:
            self.tello.go_xyz_speed(0, distance, 0, 30)
            time.sleep(3.0)
        except Exception as e:
            print(e)
            pass

    def move_right(self,distance):
        print("Moving right")
        try:
            self.tello.go_xyz_speed(0, -distance, 0, 30)
            time.sleep(3.0)
        except Exception as e:
            print(e)
            pass

    def move_up(self,distance):
        try:
            self.tello.go_xyz_speed(0, 0, distance, 30)
            time.sleep(0.05)
        except Exception as e:
            print(e)
            pass

    def move_down(self,distance):
        try:
            self.tello.go_xyz_speed(0, 0, -distance, 30)
            time.sleep(0.05)
        except Exception as e:
            print(e)
            pass

    # def align_drone_with_gap(self,center):
    #     x,y = center

    #     if x > 480:
    #         self.tello.send_rc_control(10,0,0,0)
    #     elif x < 480:
    #         self.tello.send_rc_control(-10,0,0,0)
    #     time.sleep(2)
    #     self.tello.send_rc_control(0,0,0,0)

    def find_best_optical_move_time(self):
        move_time = 0.3
        while move_time < 2:
        
            with self.frame_lock:
                self.of_frame_1 = self.most_recent_frame

            self.move_right(move_time, 20)

            with self.frame_lock:
                self.of_frame_2 = self.most_recent_frame
            
            found_gap,center_of_gap,flow_image,mag_image, mask_image = self.of_ins.find_gap_center(self.of_frame_2,self.of_frame_1)
            right_image_path = "output/right_{:.2f}".format(move_time) + ".png"

            cv2.imwrite(right_image_path, mask_image)
            image_1 = cv2.resize(self.of_frame_2, (480,360))
            image_2 = cv2.resize(flow_image, (480,360))
            image_3 = cv2.resize(mask_image, (480,360))
            image_h = cv2.hconcat([image_1,image_2, image_3])
            self.frame_handler.publish_frame(image_h)
            self.frame_handler.publish_frame(image_h)
            

            self.of_frame_1 = self.of_frame_2
            self.move_left(move_time, 20)
            with self.frame_lock:
                self.of_frame_2 = self.most_recent_frame
            
            found_gap,center_of_gap,flow_image, mask_image = self.of_ins.find_gap_center(self.of_frame_2,self.of_frame_1)
            left_image_path = "output/left_{:.2f}".format(move_time) + ".png"

            cv2.imwrite(left_image_path, mask_image)
            image_1 = cv2.resize(self.of_frame_2, (480,360))
            image_2 = cv2.resize(flow_image, (480,360))
            image_3 = cv2.resize(mask_image, (480,360))
            image_h = cv2.hconcat([image_1,image_2, image_3])
            self.frame_handler.publish_frame(image_h)
            self.frame_handler.publish_frame(image_h)
            move_time += 0.1
            print("move_time ", move_time)
            i += 1 



    
    def align_gap_horizontal(self):
        # of_ins = OpticalFlow()
        # vidcap = cv2.VideoCapture('./images/drone1.mp4')
        # success,image_1 = vidcap.read()
        # success,image_2 = vidcap.read()
        center_of_gap = None
        center_aligned = False
        move_time = 1.0
        move_speed = 20
        distance = 20
        while not center_aligned:
            with self.frame_lock:
                self.of_frame_1 = self.most_recent_frame
            time.sleep(0.5)
            with self.frame_lock:
                self.of_frame_2 = self.most_recent_frame
            # if center_of_gap is None:
            #     self.move_right(distance)
            # else:
                # x,y = center_of_gap

                # if x > 480:
                #     self.move_right(distance)
                # elif x < 480:
                #     self.move_left(distance)
            

            # with self.frame_lock:
            #     self.of_frame_2 = self.most_recent_frame

            
            found_gap,center_of_gap,flow_image,mag_image, mask_image = self.of_ins.find_gap_center(self.of_frame_2,self.of_frame_1)
            x,y = center_of_gap

            threshold = 30
            frame_to_viz = self.of_frame_2
            cv2.rectangle(frame_to_viz, (480 - threshold,360 - threshold), (480 + threshold,360 + threshold), (255, 0, 0, 255), thickness=2)
            image_1 = cv2.resize(self.of_frame_2, (480,360))
            image_2 = cv2.resize(flow_image, (480,360))
            image_3 = cv2.resize(mag_image, (480,360))
            image_4 = cv2.resize(mask_image, (480,360))
            image_h = cv2.hconcat([image_1,image_2,image_3, image_4])
            self.frame_handler.publish_frame(image_h)
            self.frame_handler.publish_frame(image_h)
            if found_gap:
                if x > 480:
                    self.move_right(distance)
                elif x < 480:
                    self.move_left(distance)

                dist_to_center = abs(480 - center_of_gap[0])
                if dist_to_center < threshold:
                    print("Aligned horizontally")
                    center_aligned = True
            else:
                print("no gap found")
                # self.move_forward(20)
    

    
    def align_gap_vertical(self):
        # of_ins = OpticalFlow()
        # vidcap = cv2.VideoCapture('./images/drone1.mp4')
        # success,image_1 = vidcap.read()
        # success,image_2 = vidcap.read()
        center_of_gap = None
        center_aligned = False
        while not center_aligned:
            with self.frame_lock:
                self.of_frame_1 = self.most_recent_frame
            
            if center_of_gap is None:
                self.move_up(2, 20)
            else:
                x,y = center_of_gap

                if y > 360:
                    self.move_down(2, 20)
                elif y < 360:
                    self.move_up(2, 20)
            
            self.tello.send_rc_control(0,0,0,0)

            with self.frame_lock:
                self.of_frame_2 = self.most_recent_frame

            
            found_gap,center_of_gap,flow_image,mag_image, mask_image = self.of_ins.find_gap_center(self.of_frame_2,self.of_frame_1)

            threshold = 50
            frame_to_viz = self.of_frame_2
            cv2.rectangle(frame_to_viz, (480 - threshold,360 - threshold), (480 + threshold,360 + threshold), (255, 0, 0, 255), thickness=2)
            image_1 = cv2.resize(self.of_frame_2, (480,360))
            image_2 = cv2.resize(flow_image, (480,360))
            image_3 = cv2.resize(mag_image, (480,360))
            image_4 = cv2.resize(mask_image, (480,360))
            image_h = cv2.hconcat([image_1,image_2,image_3, image_4])
            self.frame_handler.publish_frame(image_h)
            self.frame_handler.publish_frame(image_h)
            dist_to_center = abs(360 - center_of_gap[1])
            if dist_to_center < threshold:
                print("Aligned vertically")
                center_aligned = True

        





try:
    # Initialize the drone state machine
    drone_sm = FlyTello()

    frame_1 = cv2.imread("images/frame1.png")
    frame_2 = cv2.imread("images/frame2.png")
    found_gap,center_of_gap,flow_image,mag_image, mask_image = drone_sm.of_ins.find_gap_center(frame_1,frame_2)
    # drone_sm.frame_handler.publish_frame(mask_image)

    drone_sm.connect_tello()

    # while True:
    #     time.sleep(0.1)
    drone_sm.attempt_take_off()

    time.sleep(1)

    drone_sm.move_to_rendezvous()

    

    drone_sm.align_gap_horizontal()

    try:
        drone_sm.tello.move_forward(250)
        time.sleep(0.05)
    except Exception as e:
        print(e)
        pass

    # drone_sm.align_gap_vertical()
    # drone_sm.find_best_optical_move_time()

    try:
        drone_sm.tello.land()
    except Exception as e:
        print(e)
        pass
    print("Drone has landed successfully.")

except Exception as e:
    print(e)
    for i in range(5):
        drone_sm.tello.emergency()
    # drone_sm.abort_mission()
    drone_sm.tello.emergency()