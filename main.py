from paramiko import SSHClient, AutoAddPolicy
from picamera2 import Picamera2
from lane_follower import lane_det
# from road_signs import detect
import time
import traceback
import matplotlib.pyplot as plt
speed = 65

def give_heading(speed, turn):
    stdin.write("d")
    stdin.write("\n")
    stdin.write(str(speed))
    stdin.write("\n")
    stdin.write(str(turn))
    stdin.write("\n")

ip_add = '192.168.1.21' # of EV3 Brick
client = SSHClient()
client.load_host_keys("/home/pranjal/.ssh/known_hosts") 
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect('192.168.1.21', username='robot', password='maker')
stdin, stdout, stderr = client.exec_command('brickrun -r -- pybricks-micropython self_driving_pi/move.py')
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
picam2.configure(camera_config)
picam2.start()
time.sleep(3)
give_heading(speed, 0) 
time.sleep(1)
print("starting")
headings = []
# time = []
try:
    heading = 0
    while 1:
        picam2.capture_file('frame.jpg')
        # print("image captured")
        heading = lane_det()
        prev_h = heading
        headings.append(heading)
        # time.append(time.time())
        print("heading", heading)
        give_heading(speed, int(heading))

except:
    print("\nheadings:\n", headings)
    stdin.write("s")
    stdin.write("\n")
    client.close()
    picam2.close()
    traceback.print_exc()
