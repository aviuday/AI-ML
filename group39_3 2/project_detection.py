# A Haar Cascade is an object detection method used to locate an object of interest in images
# We will use a pre-trained Haar Cascade model from OpenCV and Python to detect and extract faces from an image.
import cv2
import glob

for i in range(15):
    img_name = "face" + str(i+1)
    for img in glob.glob("Data/Celeb_Faces_New/" + img_name + "/*.jpg"):
        img_name = "MOD_" + img[img.rfind("/") + 1:]
        imagePath = img
        # convert input image to greyscale, because detecting luminance 
        # as opposed to color, will generally yield better results in object detection
        image = cv2.imread(imagePath) # converting image to open cv object
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # input image object to a grayscale
        # create face cascade objectthat will load haar cascade file with cv2 classifier method
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # detectMultiScale() generates a list of rectangles of all detected faces in the image
        # list of rectangles is a collection of pixel locations from the image, in the form of Rect(x,y,w,h).
        faces = faceCascade.detectMultiScale( 
                gray, 
                scaleFactor=1.3, # rate to reduce the image size at each image scale
                minNeighbors=3, # how many neighbors, or detections, each candidate rectangle should have to retain it
                minSize=(30, 30) # minimum possible object size, Objects smaller than this are ignored.
        ) 
        # print(faces, "faced")
        if not len(faces): # empty tuple
            print("Classifier could not work on this image",img)
            continue
        else: 
            (x, y, w, h) = faces[0]
        # draw rectangle around faces
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) # (,,colour of line, thickness of line)
        roi_color = image[y:y + h, x:x + w] #image is image with rectangle drawn on it
        img_name = "face" + str(i+1) + img_name
        status = cv2.imwrite("Data/Celebs_15_detected/face"+str(i+1)+"/" + img_name, roi_color) 
        assert status == True
    # print("OUT OF LOOP")


