import cv2
import re
TEST_DIR_PATH = "./test/"
TRAIN_DIR_PATH = "./train/"
def generate_alphabet(dir_path, image_path, name, step, size):
	images = []
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	h, w = image.shape[:2]
	alpha_count = 0
	for i in range(0, w, step):# 180
		for j in range(0, w, step): # 180
			cropped = image[i : i + size, j : j + size] 
			try:
				resized_image = cv2.resize(cropped, (32, 32)) 
			except:
				break
			# cv2.imshow("Original image", cropped)
			# cv2.waitKey(0)
			cv2.imwrite(dir_path + name + "_{}.png".format(alpha_count) , resized_image)
			alpha_count += 1
def main():
	generate_alphabet(TRAIN_DIR_PATH, "e.png", "e", 179, 120)
	generate_alphabet(TRAIN_DIR_PATH, "u.png", "u", 59, 40)#generate alphabet images on your local disk
	generate_alphabet(TRAIN_DIR_PATH, "ia.png", "ia", 179, 120)#generate alphabet images on your local disk
	
	generate_alphabet(TEST_DIR_PATH, "test_e.png", "e", 179, 120)#generate alphabet images on your local disk
	generate_alphabet(TEST_DIR_PATH, "test_u.png", "u", 179, 120)#generate alphabet images on your local disk
	generate_alphabet(TEST_DIR_PATH, "test_ia.png", "ia", 179, 120)#generate alphabet images on your local disk

if __name__ == "__main__":
	main()