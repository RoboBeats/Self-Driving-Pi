from paramiko import SSHClient, AutoAddPolicy
import time
import traceback
import shutil
import os
from imutils.video import VideoStream
import cv2
from gpiozero import DistanceSensor

def give_heading(speed, turn):
    print("===============\n\n", int(turn))
    stdin.write("d")
    stdin.write("\n")
    stdin.write(str(int(speed)))
    stdin.write("\n")
    stdin.write(str(int(turn)))
    stdin.write("\n")

# ip_add = "192.168.68.61"
ip_add = '192.168.43.124' # ip asddress of EV3 Brick
client = SSHClient()
client.load_host_keys("/home/pranjal/.ssh/known_hosts") 
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(ip_add, username='robot', password='maker')
stdin, stdout, stderr = client.exec_command('brickrun -r -- pybricks-micropython self_driving_pi/main.py')

webcam=VideoStream(src=0).start()
os.system("v4l2-ctl -d /dev/video0 -c auto_exposure=3")
# os.system("v4l2-ctl -d /dev/video0 -c exposure_time_absolute=40")
webcam.read()
print("starting")
headings = []
time.sleep(1)
ultrasonic = DistanceSensor(trigger=23, echo=24)

from heading import get_heading

try:
    shutil.rmtree("frames/", ignore_errors=True)
    os.mkdir("frames")
    prev_heading=0
    heading = 0
    i=0
    while 1:
        dist = ultrasonic.distance
        dist *= 100
        print(dist)
        if dist < 30:
            stdin.write("s")
            stdin.write("\n")
            time.sleep(0.5)
            continue
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