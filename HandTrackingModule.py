import cv2
import mediapipe as mp
import time #for frame rate


class handDetector():
    # Constructor
    def __init__(self, mode=False, max_hands = 2, detection_conf = 0.5, track_conf = 0.5, draw = True, show_fps = True):
        self.mode = mode
        self.maxHands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.draw = draw
        self.show_fps = show_fps
        
        ###########
        # Variables
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detection_conf,self.track_conf)
        self.draw_util = mp.solutions.drawing_utils
        self.results = None

        self.previous_time = 0
        self.current_time = 0

    #######################
    # Class Functions

    #Function to update current and previous time and calculate FPS
    def get_fps(self,previous_time, current_time):
        fps = 1/(current_time - previous_time)   
        self.previous_time = current_time 
        return fps

    # Main function to detect and draw on hands
    def find_Hands(self,img, draw = True):

        num_of_hands = 0
        #Pass our image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb) #this will process the frame and return results 
        # print(results.multi_hand_landmarks) # Check if something is detected 
        #Extract info 
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                num_of_hands+=1
                if self.draw:
                    self.draw_util.draw_landmarks(img , hand, self.mpHands.HAND_CONNECTIONS)    
        if self.show_fps:
            return self.display_fps(img) , num_of_hands
        
        return (img , num_of_hands)

    def find_poisiton(self, img, hand_Num = 0): 
        
        landmark_List = []
        
        # Extract Hand ID and Landmark Information
        if self.results.multi_hand_landmarks:
            #Get hand with specific ID
            myHand = self.results.multi_hand_landmarks[hand_Num]
            # Iterate over landmarks on that hand
            for id, landmark in enumerate(myHand.landmark):
                #print(id, landmark)
                # Get heigh-width-chanels of image
                img_height,img_width,img_channels = img.shape
                # Get center of img coordinates
                center_x, center_y = int(landmark.x * img_width) , int(landmark.y * img_height)
                landmark_List.append([id,center_x,center_y])
                # Detect specific points out of the 21 - in this case , just finger tips
                if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and self.draw:
                    cv2.circle(img,(center_x,center_y), radius=7, color=(255,0,252), thickness=cv2.FILLED)

        return landmark_List

    def display_fps(self,img):
        fps = self.get_fps(self.previous_time,time.time())
        #Display FPS on screen
        cv2.putText(img, str(int(fps)), org = (10,40), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0,20,255), thickness=2)
        return img
#####################################################
# Main Function
def main():
    #initialise video capture object (webcam)
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.find_Hands(img)
        position = detector.find_poisiton(img)
        img = detector.display_fps(img)

        #Show image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()