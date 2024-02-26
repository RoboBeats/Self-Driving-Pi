from paramiko import SSHClient, AutoAddPolicy
from picamera2 import Picamera2
from heading import get_heading
import time
import traceback
import shutil
import os

def give_heading(speed, turn):
    stdin.write("d")
    stdin.write("\n")
    stdin.write(str(int(speed)))
    stdin.write("\n")
    stdin.write(str(int(turn)))
    stdin.write("\n")

ip_add = 'ev3dev.local' # ip address of EV3 Brick
client = SSHClient()
client.load_host_keys("/home/pranjal/.ssh/known_hosts") 
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(ip_add, username='robot', password='maker')
stdin, stdout, stderr = client.exec_command('brickrun -r -- pybricks-micropython self_driving_pi/main.py')

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
give_heading(70, 0)

print("starting")
headings = []
try:
    shutil.rmtree("frames/", ignore_errors=True)
    os.mkdir("frames")
    prev_heading=0
    heading = 0
    i=0
    while 1:
        picam2.capture_file(f'frames/frame_{i}.jpg')

        speed, heading = get_heading(prev_heading, f"frames/frame_{i}.jpg")
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
    picam2.close()
    traceback.print_exc()
