import cv2
import time
import torch
import numpy as np
import argparse

from FER.models.CnnModule import CNNModule
RED_COLOR_CODE = (0, 0, 255)
CLASSES = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--input', default='webcam')
    parser.add_argument('--output', default=None)

    return parser.parse_args()

def drawBoundingBoxes(imageData, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    thick = 1
    imgHeight, _, _ = imageData.shape
    for res in inferenceResults:
        x, y, w, h = res['pos']
        label = res['label']
        label = res['label']
        cv2.rectangle(imageData,(x, y), (x+w, y+h), color, thick)
        cv2.putText(imageData, label, (x, y - 12), 0, 1e-3 * imgHeight, color, thick)
    return imageData



class VideoCapture():
    def __init__(self, input_path = None, output_path = None):
        if input_path == 'webcam':
            self.stream = cv2.VideoCapture(0)
        else:
            self.stream = cv2.VideoCapture(input_path)
        if (self.stream.isOpened() == False): 
            print("Unable to read camera feed")
            return
        
        if output_path != None:
            self.write_mode = 'video'
            frame_width = int(self.stream.get(3))
            frame_height = int(self.stream.get(4))
            
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        else:
            self.write_mode = 'screen'
    
    def read(self):
        ret, frame = self.stream.read()
        return ret, frame
    
    def isOpened(self):
        return self.stream.isOpened()

    def write(self, frame):
        if self.write_mode == 'screen':
            cv2.imshow('frame',frame)
        else:
            self.out.write(frame)
    
    def release(self):
        self.stream.release()
        cv2.destroyAllWindows()
        if self.write_mode == 'video':
            self.out.release()

class FaceRecognition():
    def __init__(self, haarcascade_path, ckpt_path, label_names, device='cpu') -> None:
        
        self.detector = cv2.CascadeClassifier(haarcascade_path)
        
        self.classifier = CNNModule.load_from_checkpoint(ckpt_path).to(device)
        torch.set_grad_enabled(False)
        self.classifier.eval()
        
        self.device = device
        self.face_size = (64, 64)
        self.label_names = label_names
    
    def preprocess_detector(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def preprocess_classifier(self, img):
        img = cv2.resize(img, self.face_size)
        img = img / 255.0
        img = torch.from_numpy(img).to(self.device).float()
        return img
    
    def __call__(self, img):
        img = self.preprocess_detector(img)
        results = self.detector.detectMultiScale(img)
        poss = []
        faces = []
        for (x, y, w, h) in results:
            face = img[y:y+h, x:x+w]
            face = self.preprocess_classifier(face)
            faces.append(face)
            poss.append((x, y, w, h))
        
        if len(faces) == 0:
            return []
    
        faces = torch.stack(faces)
        faces = faces.unsqueeze(1)
        labels = self.classifier(faces)
        labels = torch.argmax(labels, 1)

        labels = [self.label_names[labels[i]] for i in range(len(labels))]
        result_wlabels = [dict(pos=pos, label=label) for pos, label in zip(poss, labels)]

        return result_wlabels


def build_recognizer(args):
    cascade_path = 'haarcascade/haarcascade_frontalface_default.xml'
    ckpt_path = 'checkpoints/Best_Checkpoint-v3.ckpt'
    recognizer = FaceRecognition(cascade_path, ckpt_path, CLASSES, torch.device(args.device))
    return recognizer

def main():
    args = parse_args()
    recognizer = build_recognizer(args)  
  
    # define a video capture object
    read_path = args.input
    write_path = args.output
    vid = VideoCapture(read_path, write_path)
    
    while(True):
        start_time = time.time() # start time of the loop
        
        ret, frame = vid.read()
        if ret == False:
            break
       
        results = recognizer(frame)
        #################
        
        # WRITE #########
        frame = drawBoundingBoxes(frame, results, RED_COLOR_CODE)
        vid.write(frame)
        #################
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    
    vid.release()
if __name__ == '__main__':
    main()