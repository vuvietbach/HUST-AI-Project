import cv2
import time
RED_COLOR_CODE = (0, 0, 255)

def drawBoundingBoxes(imageData, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['left']) + int(res['width'])
        bottom = int(res['top']) + int(res['height'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
    return imageData


def inference(frame, face_detector, classifier):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_detector(frame)

    for (x, y, w, h) in faces:
        # Extract image
        positions = None

        # Normalize
        # Batching
        images=None
    labels = classifier(images)
    labels = np.argmax(labels, 1)
    # Extract string name of labels
    labels = None
    return positions, labels


class VideoCapture():
    def __init__(self, read_path = None, write_path = None):
        if read_path == None:
            self.stream = cv2.VideoCapture(0)
        else:
            self.stream = cv2.VideoCapture(read_path)
            
            if (self.stream.isOpened() == False): 
                print("Unable to read camera feed")
                return
        
        if write_path != None:
            self.write_mode = 'video'
            frame_width = int(self.stream.get(3))
            frame_height = int(self.stream.get(4))
            
            self.out = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        else:
            self.write_mode = 'screen'
    
    def read(self):
        ret, frame = self.stream.read()
        return ret, frame
    
    def isOpened(self):
        return self.stream.isOpened()

    def write(self, frame):
        if self.mode == 'screen':
            cv2.imshow('frame',frame)
        else:
            self.out.write(frame)
    
    def release(self):
        self.stream.release()
        cv2.destroyAllWindows()
        if self.mode == 'video':
            self.out.release()

    

def main():
    device='cuda:3'
    # define model
    model = CustomCNNModule.load_from_checkpoint(ckpt_path).to(device)
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    # load checkpoint
  
  
    # define a video capture object
    vid = VideoCapture()
    
    while(True):
        start_time = time.time() # start time of the loop
        ## READ ##########
        ret, frame = vid.read()
        if ret == False:
            break
        #################
       
        # INFERENCE #####
        inferenceResults = inference(frame)
        #################
        
        # WRITE #########
        frame = drawBoundingBoxes(frame, inferenceResults, RED_COLOR_CODE)
        vid.write(frame)
        #################
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    
    vid.release()
if __name__ == '__main__':
    main()