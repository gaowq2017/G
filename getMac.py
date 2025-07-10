import cv2
import os

image_path = r'D:\data\data\images'
template_path = r'D:\data\data\iPad_Template\iPadTemplateHand.jpg'
save_path = r'D:\data\data\save'
image_template = cv2.imread(template_path)
file_list = os.listdir(image_path)
for file in file_list:
    file_path = os.path.join(image_path, file)
    image = cv2.imread(file_path)
    res = cv2.matchTemplate(image, image_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    image_start = (max_loc[0], max_loc[1])
    image_end = (image_start[0] + image_template.shape[1], image_start[1] + image_template.shape[0])
    image_iPad = image[image_start[1]:image_end[1], image_start[0]:image_end[0], :]
    image_save_path = os.path.join(save_path, file)
    cv2.imwrite(image_save_path, image_iPad)


@server.route('GDetectiPad', methods=['get', 'post'])
def getiPad():
    inputPath = flask.request.args.get("inputPath")
    outputPath = flask.request.args.get("outputPath")
    try:
        file = os.path.basename(inputPath)
        image = cv2.imread(inputPath)
        res = cv2.matchTemplate(image, IPADTEMPLATE, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        image_start = (max_loc[0], max_loc[1])
        image_end = (image_start[0] + IPADTEMPLATE.shape[1], image_start[1] + IPADTEMPLATE.shape[0])
        image_iPad = image[image_start[1]:image_end[1], image_start[0]:image_end[0], :]
        image_save_path = os.path.join(outputPath, file)
        cv2.imwrite(image_save_path, image_iPad)
        ##### Detect iPad
        gray = cv2.cvtColor(image_iPad, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        width = gray.shape[1]
        intersection_area = height * width
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_box = None
        max_area = 0
        epsilon = 0.02
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_box = contour

        if largest_box is not None:
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest_box], -1, 255, thickness=cv2.FILLED)
            box_area = cv2.contourArea(largest_box)
            ritio = box_area / intersection_area
            if ritio > 0.6:
                res = {'msg': 'OK'}
            else:
                res = {'msg': 'NG'}
        else:
            res = {'msg': 'OK'}
    except:
        res = {'msg': 'TimeOut'}

    return json.dumps(res, ensure_ascii=False)
