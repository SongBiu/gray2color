import random
import os
import shutil

n = 1815
imgs = os.listdir("64/gray")
ret = []

for i in range(n):
	item = random.choice(imgs)
	while item in ret:
		item = random.choice(imgs)
	ret.append(item)
# print(ret)

for item in ret:
	source_path_gray = "64/gray/%s" % item
	source_path_color = "64/color/%s" % item

	target_path_gray = "64/test/gray/%s" % item
	target_path_color = "64/test/color/%s" % item
	shutil.move(source_path_gray, target_path_gray)
	shutil.move(source_path_color, target_path_color)