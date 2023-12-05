from paramiko import SSHClient, AutoAddPolicy
from picamera2 import Picamera2
from heading import get_heading
# from road_signs import detect
import time
import traceback
import matplotlib.pyplot as plt
speed = 60

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
client.connect('192.168.1.5', username='robot', password='maker')
stdin, stdout, stderr = client.exec_command('brickrun -r -- pybricks-micropython self_driving_pi/move.py')

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
picam2.configure(camera_config)
picam2.start()
time.sleep(3)
give_heading(0, 0) 
time.sleep(1)

print("starting")
headings = []
try:
    prev_heading=0
    heading = 0
    i=0
    while 1:
        picam2.capture_file(f'frame.jpg')
        prev_heading = heading
        heading = get_heading(prev_heading, img_name=f"frame.jpg")
        i += 1
        headings.append(heading)
        print("\n=======================> heading", heading, "\n")
        give_heading(speed, int(heading))
        prev_heading = heading

except:
    print("\nheadings:\n", headings)
    stdin.write("s")
    stdin.write("\n")
    client.close()
    picam2.close()
    traceback.print_exc()
