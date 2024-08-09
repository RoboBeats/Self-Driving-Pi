from gpiozero import DistanceSensor

ultrasonic = DistanceSensor(trigger=23, echo=24)

while True:
    print(ultrasonic.distance)
