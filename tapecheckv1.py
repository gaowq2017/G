import cv2
import yaml
from tqdm import tqdm
import glob
import time
import os, re
import numpy as np
import PIL.Image
import PIL.ImageDraw 
import math
import torch
import torch.nn as nn
from torchvision import transforms


class ReadConfig():
    def __init__(self, config_file):
        """
        Args:
            config_file (str): config file path
        """
        self.readConfigFile(config_file)

    def readConfigFile(self, file):
        try:
            with open(file, errors='ignore') as f:
                config = yaml.safe_load(f)
            for key in config:
                setattr(self, key, config[key])
        except:
            print('Configuration file read error')
            raise


class TapePrediction(ReadConfig):
    def __init__(self, config_file):

        super().__init__(config_file)
        self.PRED_IMGPATH = self.PRED_IMGPATH
        self.RESULT_PATH = self.RESULT_PATH
        self.template = cv2.imread(self.template_path)
        self.template_img = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        self.nominal = np.array([self.angle, self.size, self.x, self.y])
        self.spec = np.array([self.angle_spec, self.size_spec, self.x_spec, self.y_spec])
        self.idx1 = self.idx1
        self.idx2 = self.idx2
        self.idx3 = self.idx3
        self.idx4 = self.idx4
        self.Cnt = np.array(self.Cnt)
        self.Pnt = np.array(self.Pnt)
        self.Length = self.Length
        self.Width = self.Width
    
        Template_path = self.pattern_path
        ## Load FOV related template
        self.Tsep = []
        i = 0
        T1 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T1.jpg', cv2.IMREAD_GRAYSCALE)
        T01 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T01.jpg', cv2.IMREAD_GRAYSCALE)
        T2 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T2.jpg', cv2.IMREAD_GRAYSCALE)
        T02 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T02.jpg', cv2.IMREAD_GRAYSCALE)

        M1, M01, M2, M02 = [], [], [], []
        if os.path.exists(Template_path +'/FOV'+str(i+1)+'/T1.png'):
            M1 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T1.png', cv2.IMREAD_GRAYSCALE)
            M01 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T01.png', cv2.IMREAD_GRAYSCALE)
        if os.path.exists(Template_path +'/FOV'+str(i+1)+'/T2.png'):
            M2 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T2.png', cv2.IMREAD_GRAYSCALE)
            M02 = cv2.imread(Template_path +'/FOV'+str(i+1)+'/T02.png', cv2.IMREAD_GRAYSCALE)
        self.Tsep.append([T1, T2, T01, T02, M1, M2, M01, M02])
    
    def GetMatchTemplate(self, img0, idxtt1, idxtt2, T):
        n_point = 5
        f = nn.AvgPool2d(2, stride=2)
        trans = transforms.Compose([transforms.ToTensor()])
        # print(idxtt1, idxtt2)
        imgt1 = img0[idxtt1[0]:idxtt1[1]-1, idxtt2[0]:idxtt2[1]-1]
        imgt01 = trans(imgt1)
        imgt01 = imgt01[None]
        
        ## scaling image to speed up
        for i in range(2):
            imgt01 = f(imgt01)
        imgt01 = torch.squeeze(imgt01).numpy()*255
        imgt01 = imgt01.astype(np.uint8)

        T1,T01,M1,M01 = T[0], T[1], T[2], T[3]
        ## do template match roughly
        res = cv2.matchTemplate(imgt01, T01, cv2.TM_CCOEFF_NORMED) if len(M01) == 0 else cv2.matchTemplate(imgt01, T01, cv2.TM_CCOEFF_NORMED,mask=M01)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        k1, h1 = max_loc[0]*4, max_loc[1]*4

        xs1 = k1-n_point if k1 > n_point else 0
        ys1 = h1-n_point if h1 > n_point else 0

        xe1 = k1+T1.shape[1]+n_point if k1+T1.shape[1]+n_point < imgt1.shape[1] else imgt1.shape[1]
        ye1 = h1+T1.shape[0]+n_point if h1+T1.shape[0]+n_point < imgt1.shape[0] else imgt1.shape[0]

        imgt1 = imgt1[ys1:ye1, xs1:xe1]
        ## do pixel level template match
        res = cv2.matchTemplate(imgt1, T1, cv2.TM_CCOEFF_NORMED) if len(M1) == 0 else cv2.matchTemplate(imgt1, T1, cv2.TM_CCOEFF_NORMED, mask=M1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        k1, h1 = max_loc[0], max_loc[1]

        return xs1+k1, ys1+h1, max_val

    def PointsConvert(self, P12, P12t, Pnt):
        
        dx = np.mean(P12[:, 0] - P12t[:, 0])
        dy = np.mean(P12[:, 1] - P12t[:, 1])

        theta1 = math.atan2(P12[1, 1] - P12[0, 1], P12[1, 0] - P12[0, 0])
        theta2 = math.atan2(P12t[1, 1] - P12t[0, 1], P12t[1, 0] - P12t[0, 0])
        dtheta = theta2 - theta1

        kt = math.sqrt(np.sum((P12[0, :] - P12[1, :])**2))/math.sqrt(np.sum((P12t[0, :] - P12t[1, :])**2))

        Pnt1 = Pnt + [dx, dy]

        Cnt = np.mean(P12, axis=0)
        Pnt1 = Pnt1 - Cnt
        Pnt2 = np.array([Pnt1[:,0]*math.cos(dtheta)+Pnt1[:, 1]*math.sin(dtheta), -Pnt1[:,0]*math.sin(dtheta)+Pnt1[:, 1]*math.cos(dtheta)]).T
        Pnt3 = kt*Pnt2 + Cnt

        return Pnt3[0], Pnt3[1], Pnt3[2], Pnt3[3], Pnt3
    
    def imgmerge(self, path):

        # img_list0 = [path]
        
        idx = 0

        # pattern = 'FOV'+str(idx+1)
        # img_list = list(filter(lambda x: re.search(pattern, x)!=None, img_list0))

        # img_list = [s for s in img_list if not s.startswith('.')]
        # img_file = img_list[0]
        image = PIL.Image.open(path)

        T1, T2, T01, T02 = self.Tsep[idx][0], self.Tsep[idx][1], self.Tsep[idx][2], self.Tsep[idx][3]
        M1, M2, M01, M02 = self.Tsep[idx][4], self.Tsep[idx][5], self.Tsep[idx][6], self.Tsep[idx][7]
        idxt1, idxt2, idxt3, idxt4 = self.idx1[idx], self.idx2[idx], self.idx3[idx], self.idx4[idx]

        x01, y01, x02, y02 = T1.shape[1]/2, T1.shape[0]/2, T2.shape[1]/2, T2.shape[0]/2

        cnt, pnt = self.Cnt[idx], self.Pnt[idx]
        L, W = self.Length[idx], self.Width[idx]

        ######## do template match
        img1 = np.array(image)
        img0 = img1[:, :, 0]
        
        # print(idxt1, idxt2)
        
        k1, h1, matchscore1 = self.GetMatchTemplate(img0, idxt1, idxt2, [T1,T01,M1,M01])
        k2, h2, matchscore2 = self.GetMatchTemplate(img0, idxt3, idxt4, [T2,T02,M2,M02])

        Cnt1 = np.array((k1+T1.shape[1]/2-1+idxt2[0]-1, h1+T1.shape[0]/2-1+idxt1[0]-1))
        Cnt2 = np.array((k2+T2.shape[1]/2-1+idxt4[0]-1, h2+T2.shape[0]/2-1+idxt3[0]-1))
        ######## Convertion
        P12 = np.array([Cnt1, Cnt2])

        Pt3, Pt4, Pt5, Pt6, _ = self.PointsConvert(P12, cnt, pnt)
        #######################
        # points1 = np.float32([Pt5, Pt6, Pt3, Pt4])
        points1 = np.float32([Pt3, Pt4, Pt5, Pt6])
        points2 = np.float32([[0, 0], [0, W-1], [L-1, W-1], [L-1, 0]])

        M = cv2.getPerspectiveTransform(points1, points2)
        img2 = cv2.warpPerspective(img1, M, (L, W))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        return img2    

    def predict(self, raw_img_path):
        
        raw_img_color = self.imgmerge(raw_img_path)
        
        raw_img = cv2.cvtColor(raw_img_color, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors
        keypoints_main, descriptors_main = self.sift.detectAndCompute(raw_img, None)
        keypoints_template, descriptors_template = self.sift.detectAndCompute(self.template_img, None)

        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors
        matches = flann.knnMatch(descriptors_template, descriptors_main, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Get matched keypoints
        matched_points_template = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matched_points_main = np.float32([keypoints_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix and perspective transform
        H, _ = cv2.findHomography(matched_points_template, matched_points_main, cv2.RANSAC, 5.0)

        # Apply perspective transform to the corners of the template image
        height, width = self.template_img.shape
        corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        # Calculate angle and matching size
        angle = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
        matching_size = np.linalg.norm(H[:, 0])
        
        centercor = np.average(transformed_corners, axis=1)
        centercor = np.round(np.average(centercor, axis=0)).astype(int)

        # Draw_img the matches, bounding box, angle, and size
        # main_with_matches = cv2.drawMatches(template_img, keypoints_template, raw_img, keypoints_main, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        main_with_bb = cv2.polylines(raw_img_color, [np.int32(transformed_corners)], True, (0, 255, 0), 10)
        main_with_bb = cv2.circle(main_with_bb, centercor, radius=20, color=(0, 255, 0), thickness=-1)
        # cv2.putText(main_with_bb, f'Angle: {angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(main_with_bb, f'Matching Size: {matching_size:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        # cv2.imshow('Bounding Box', main_with_bb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()   
        
        characters = ['Angle NG ', 'Missing ', 'X Position NG ', 'Y Position NG ']
        
        data = np.array([angle, matching_size, centercor[0], centercor[1]])
        
        print(data)
        
        diff = self.spec - np.abs(data - self.nominal)

        message = ''
        result = True

        # Iterate through the array and characters simultaneously
        for value, char in zip(diff, characters):
            # Check if the value is less than 0
            if value < 0:
                # Append the corresponding character to the final string
                message += char
                result = False
        
        rotated_image = cv2.rotate(main_with_bb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.putText(rotated_image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
        cv2.imwrite(self.RESULT_PATH+'/result.jpg', rotated_image)
        
        return result


def main():
    time_tracker = {}
    time_sum = 0

    predictor = TapePrediction('/Users/stevexiong/Desktop/iPhone AOI Check/todeploy/config.yaml')

    # process each image in folder
    assert os.path.exists(
        predictor.PRED_IMGPATH), f'Image folder {predictor.PRED_IMGPATH} does not exist'
    
    # for imgPath in tqdm(glob.glob(predictor.PRED_IMGPATH + '/**/' + '*.jpg', recursive=True)):
    for imgPath in tqdm(glob.glob(predictor.PRED_IMGPATH + '/**/' + '*.jpg', recursive=True)):
        t0 = time.time()

        
        result = predictor.predict(imgPath)
        
        print(result)

        delta_t = time.time() - t0
        time_tracker[imgPath] = delta_t

    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum / len(time_tracker)))


if __name__ == "__main__":
    main()
