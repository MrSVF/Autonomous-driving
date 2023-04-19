# import os
import cv2

from PIL import Image, ImageStat
# Load image
im = Image.open('c:/Users/iserg/rebels_code/adasys_data/solid_line-bump/2023_0323_070928_F_0000000199.jpg')

# Calculate statistics
stats = ImageStat.Stat(im)                                                                 

for band,name in enumerate(im.getbands()): 
    print(f'Band: {name}, min/max: {stats.extrema[band]}, stddev: {stats.stddev[band]}')



image = cv2.imread('c:/Users/iserg/rebels_code/adasys_data/solid_line-bump/2023_0323_070928_F_0000000199.jpg')

alpha = 3 # Contrast control (1.0-3.0)
beta = -250 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow('original', image)
cv2.imshow('adjusted', adjusted)
cv2.imwrite('c:/Users/iserg/rebels_code/adasys_data/solid_line-bump/2023_0323_070928_F_0000000199_adj3-250.jpg', adjusted)
cv2.waitKey()




# import cv2
# import numpy as np

# img = cv2.imread('c:/Users/iserg/rebels_code/adasys_data/solid_line-bump/2023_0323_070928_F_0000000199.jpg', 1)
# # converting to LAB color space
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# l_channel, a, b = cv2.split(lab)

# # Applying CLAHE to L-channel
# # feel free to try different values for the limit and grid size:
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl = clahe.apply(l_channel)

# # merge the CLAHE enhanced L-channel with the a and b channel
# limg = cv2.merge((cl,a,b))

# # Converting image from LAB Color model to BGR color spcae
# enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# # Stacking the original image with the enhanced image
# result = np.hstack((img, enhanced_img))
# cv2.imshow('original', img)
# cv2.imshow('Result', result)
# cv2.waitKey()








# class Base:
#     def __init__(self, path: str, path_out: str):
#         self._GPU: str = 'cuda:0'
#         self._CPU: str = 'cpu'

#         self.path = path
#         self.path_out = path_out

#         self.model = None
#         self.device = None

#         self.weights = None
#         # self.weights_path = 'src/models/lane/weights/weights.pt'
#         self.weights_path = 'src/lane_detector_service/trained_models/2022-12-8/efficientnet-v2-s-weights-100-epochs/15-22/80_15.pt'

#         self.frames_paths = None

#         self.report = {'results': {self.weights_path: []}}
#         # self.report = {}


#     def read_frames_paths(self):
#         self.frames_paths = os.listdir(self.path)

#     @staticmethod
#     def read_frame(frame_path: str):
#         return cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)


#     def run(self):
#         self.read_frames_paths()

#         if not os.path.exists(self.path_out):
#                 os.makedirs(self.path_out)

#         for frame_path in self.frames_paths:
#             frame = self.read_frame(frame_path=os.path.join(self.path, frame_path))
#             try:
#                 # frame = torch.tensor(np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0).astype(np.float32))
#                 # print('shape1:', (frame.shape))
#                 frame = frame[0:800, 960:1920]
#                 # print('shape2:', (frame.shape))
#                 # print('path_out:', os.path.join(self.path_out, frame_path))
#                 cv2.imwrite(os.path.join(self.path_out, frame_path), frame)
#             except:
#                 pass

#         return self

# '''
# Пример команды запуска:
# python /rebel/services/adasys/github/adasys/src/models/lane/crop_images.py --img_folder=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/20200904_175240 --res_folder=/rebel/services/adasys/adasys_data/frames_retrieval_service/frames/line/20200904_175240_cr
# '''

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PytorchAutoDrive lane directory vis', conflict_handler='resolve')
#     parser.add_argument('--img_folder', type=str, help='Path to images folder', required=True)
#     parser.add_argument('--res_folder', type=str, help='Path to results folder', required=True)

#     args = parser.parse_args()

#     path_to_folder_with_frames = args.img_folder
#     path_to_folder_with_results = args.res_folder

#     runner = Base(path=path_to_folder_with_frames, path_out=path_to_folder_with_results).run()
