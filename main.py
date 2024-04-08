from paramiko import SSHClient, AutoAddPolicy
from heading import get_heading
import time
import traceback
import shutil
import os
from imutils.video import VideoStream
import cv2

def give_heading(speed, turn):
    print("===============\n\n", int(turn))
    stdin.write("d")
    stdin.write("\n")
    stdin.write(str(int(60)))
    stdin.write("\n")
    stdin.write(str(int(turn)))
    stdin.write("\n")

ip_add = '192.168.68.61' # ip address of EV3 Brick
client = SSHClient()
client.load_host_keys("/home/pranjal/.ssh/known_hosts") 
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(ip_add, username='robot', password='maker')
stdin, stdout, stderr = client.exec_command(
    'brickrun -r -- pybricks-micropython self_driving_pi/main.py')

webcam=VideoStream(src=0).start()
webcam.read()
print("starting")
headings = []
try:
    shutil.rmtree("frames/", ignore_errors=True)
    os.mkdir("frames")
    prev_heading=0
    heading = 0
    i=0
    while 1:
        img = webcam.read()
        cv2.imwrite(f'frames/frame_{i}.jpg', img)

        speed, heading = get_heading(prev_heading, f"frames/frame_{i}.jpg", stdin)
        i += 1
        headings.append(heading)
        print("\n=======================> heading", heading, "\n")
        give_heading(speed, heading)
        prev_heading = heading

except:
    print("\nheadings:\n", headings)
    stdin.write("s")
    stdin.write("\n")
    client.close()
    traceback.print_exc()
