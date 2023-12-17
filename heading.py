import sys
import PID
from create_lanes import get_lanes
from picamera2 import Picamera2

SPEED = 60
# Initialise PID
P = 0.65
I = 0.4
D = 0
pid = PID.PID(-P, I, D)
pid.SetPoint = 0

def get_heading(prev_heading, img_name = "frame.jpg"):
    lanes, stop = get_lanes(img_name, "results/")
    if stop:
        return 0, 0
    angle = 0
    if len(lanes) == 0:
        print("\n\n_________________________________\n\nNO LANES!\n_____________________")
        return SPEED, -prev_heading
    else:
        for lane in lanes:
            angle += lane[5]
        angle/= len(lanes)

    angle = 90 -angle
    print("pre pid angle: ", angle)
    pid.update(angle)
    final_heading = round(pid.output, 2)
    if -5 < final_heading < 5:
        return SPEED*1.5, final_heading*1.5
    return SPEED, final_heading

if __name__=="__main__":
    print("with camera?")
    if input()=="y":
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
        picam2.configure(camera_config)
        picam2.start()
        picam2.capture_file(f'frame.jpg')
        speed, heading = get_heading(0)
        print("speed, heading: ", speed, heading)
    else:
        speed, heading = get_heading(0, img_name=sys.argv[1:][0])
        print("speed, heading: ", speed, heading)