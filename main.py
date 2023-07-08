from paramiko import SSHClient, AutoAddPolicy
from picamera2 import Picamera2
from lane_follower import lane
# from road_signs import detect
import time

def give_heading(speed, turn):
    stdin.write("d")
    stdin.write("\n")
    stdin.write(str(speed))
    stdin.write("\n")
    stdin.write(str(turn))
    stdin.write("\n")


client = SSHClient()
client.load_host_keys("/home/pranjal/.ssh/known_hosts") 
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect('192.168.1.26', username='robot', password='maker')
stdin, stdout, stderr = client.exec_command('brickrun -r -- pybricks-micropython self_driving_pi/move.py')
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
picam2.configure(camera_config)
picam2.start()
time.sleep(3)
give_heading(75, 0) 
try:
    heading = 0
    speed = 75
    print("before while loop")
    while 1:
        picam2.capture_file('frame.jpg')
        print("image _captured")
        # detection = detect()
        # if detection >= 1:
        #     if detection > 1:
        #         speed = detection
        heading = lane(heading)
        prev_h = heading
        give_heading(speed, heading)
        # else:
        #     stdin.write("s")
        #     stdin.write("\n")
        time.sleep(0.1)

except Exception as e:
    print("\n\n")
    print(str(e))
    print("\n\n")
    stdin.write("s")
    stdin.write("\n")
    client.close()
    picam2.close()
