import os
from heading import get_heading

img_names = os.listdir("AI_data/")

headings = []
for img_name in img_names:
    heading = get_heading(0, "AI_data/"+img_name)
    headings.append(heading)

print("\n\n-----------HEADINGS--------")
for i in range(len(headings)):
    print(headings[i], img_names[i])