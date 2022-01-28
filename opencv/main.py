import cv2
import argparse
import numpy as np

class yolov5():
    def __init__(self, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        anchors = [[4, 5, 8, 10, 13, 16], [23, 29, 43, 55, 73, 105], [146, 217, 231, 300, 335, 433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNet('best.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
    def enable_dynamic_hw(self, srcimg):
        srch,srcw = srcimg.shape[:2]
        if srch>srcw:
            newh = self.inpHeight
            neww = int(self.inpHeight * (srcw/srch))
            # neww = self.inpWidth
            # newh = int(self.inpWidth * (srch/srcw))
        else:
            neww = self.inpWidth
            newh = int(self.inpWidth * (srch/srcw))
        img = cv2.resize(srcimg, (neww, newh))
        return img, (newh, neww)
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img
    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        
        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[13]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = (detection[5:13] - np.tile(np.float32([padw, padh]), 4)) * np.tile(np.float32([ratiow, ratioh]), 4)
                landmarks.append(landmark.astype(np.int32))
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            landmark = landmarks[i]
            frame = self.drawPred(frame, confidences[i], left, top, left + width, top + height, landmark)
        return frame
    
    def drawPred(self, frame, conf, left, top, right, bottom, landmark):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        # label = '%.2f' % conf
        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        for i in range(4):
            cv2.circle(frame, (landmark[i * 2], landmark[i * 2 + 1]), 2, (0, 255, 0), thickness=-1)
        return frame
    
    def detect(self, srcimg):
        # blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        img, newh, neww, top, left = self.resize_image(srcimg)
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        # img, dynamic_hw = self.enable_dynamic_hw(srcimg)
        # newh, neww, top, left = dynamic_hw[0], dynamic_hw[1], 0, 0
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        
        # Sets the input to the network
        self.net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)
        
        # inference output
        outs[..., [0, 1, 2, 3, 4, 13]] = 1 / (1 + np.exp(-outs[..., [0, 1, 2, 3, 4, 13]]))  ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)
            
            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(
                self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i
            
            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length, 5:7] * a_g_i + g_i * int(
                self.stride[i])  # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind + length, 7:9] * a_g_i + g_i * int(
                self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind + length, 9:11] * a_g_i + g_i * int(
                self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind + length, 11:13] * a_g_i + g_i * int(
                self.stride[i])  # landmark x4 y4
            row_ind += length
        srcimg = self.postprocess(srcimg, outs, padsize=(newh, neww, top, left))
        return srcimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='imgs/1.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    args = parser.parse_args()
    
    yolonet = yolov5(confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold,
                     objThreshold=args.objThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = yolonet.detect(srcimg)
    
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()