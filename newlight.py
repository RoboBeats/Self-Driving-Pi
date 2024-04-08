import cv2
import sys
import numpy as np

# Check if an image file path is provided
if len(sys.argv) < 2:
    print("Please provide an image file path as an argument.")
    sys.exit(1)

# Load the image
img_path = sys.argv[1]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
cv2.imshow("Original", img)
cv2.waitKey(0)
# Check if the image was loaded successfully
if img is None:
    print(f"Failed to load the image: {img_path}")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(img.shape)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Use the Hough Circle Transform to detect circles
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=50, minRadius=25, maxRadius=0)

# Ensure at least some circles were found
if circles is not None:
    # Convert the circle coordinates from (x, y, r) to integers
    circles = np.round(circles[0, :]).astype("int")

    # Sort the circles based on their y-coordinates
    circles = sorted(circles, key=lambda x: x[1])

    # Check if the detected circles represent a traffic signal
    if len(circles) >= 3:
        
        # Check for vertical alignment and relative sizes
        y_diffs = [circles[i][1] - circles[i-1][1] for i in range(1, len(circles))]
        r_diffs = [circles[i][2] - circles[i-1][2] for i in range(1, len(circles))]
        if True:
            #if all(abs(diff) < 20 for diff in y_diffs) and all(abs(diff) < 10 for diff in r_diffs):
            print("Detected circles represent a traffic signal")
            colors = []
            red_lower = (0, 100, 100)
            red_upper = (20, 255, 255)
            yellow_lower = (20, 100, 100)
            yellow_upper = (30, 255, 255)
            green_lower = (50, 100, 100)
            green_upper = (70, 255, 255)
            for (x, y, r) in circles:
                # Get the average color within the circle
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                mean_color = cv2.mean(img, mask=mask)[:3]
                print("mean_color: ", mean_color)
               
                # Check if the mean color falls within the defined ranges
                if cv2.inRange(mean_color, red_lower, red_upper).all() > 0:
                    color = "Red"
                elif cv2.inRange(mean_color, yellow_lower, yellow_upper).all() > 0:
                    color = "Yellow"
                elif cv2.inRange(mean_color, green_lower, green_upper).all() > 0:
                    color = "Green"
                else:
                    color = "Unknown"

                print(color)
                print("----------------------------------------")

                colors.append(color)

                # Draw the circle on the original image and display the color
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.putText(img, color, (x-r, y-r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Determine the illuminated color
            illuminated_color = max(set(colors), key=colors.count)
            print(f"The {illuminated_color} signal is currently illuminated.")
        else:
            print("Detected circles do not represent a traffic signal")
    else:
        print("Not enough circles detected for a traffic signal")

    # Display the output image
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles found")
