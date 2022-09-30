from PIL import Image
import numpy as np
import cv2
import math

# setting variables
tracking_people={}
person_id=0
frame_num = 0
input_size=416
frame_count=0
prev_frame_center_points=[]

# Reading the video file
video = cv2. VideoCapture('people_motion.avi')

# iterate frame by frame
# detect the object using yolo v4 and get the boudning boxes and classes
# find the Euclidean distance between all the objects detected in current and previous frame
while True:
        return_value, frame = video.read()
        frame_count+=1
        current_frame_center_points=[]
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        # Normalize the frame data
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        
        # detect the objects in the current frame
        pred_bbox = infer(batch_data)
        
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.5
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = format_boxes(bboxes, original_h, original_w)
                
        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = read_class_names('./data/classes/coco.names')
        
        # loop over the contours
        for cnt, obj_class  in zip(bboxes, classes):
            # if a person is detected in the frame, ignoring other objects
            if int(obj_class)==0:
                (x,y,w,h)= (int(cnt[0]), int(cnt[1]), int(cnt[2]), int(cnt[3]))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                # get the center of the objects detected
                cx=int((x+x+w)/2)
                cy=int((y+y+h)/2)
                #append the center points of all object of interest detected in the frame
                current_frame_center_points.append((cx, cy))
                
        if frame_count <=2:
            for pt in current_frame_center_points:
                for pt_prev in prev_frame_center_points:
                    distance= math.hypot(pt[0] -pt_prev[0], pt[1] -pt_prev[1])
                    # if the Euclidean distance between objects in previous frma 
                    # and current frame is less than 50 then add object for tracking
                    if distance<55:
                        tracking_people[person_id]=pt
                        person_id+=1
        else:
            # find if it is the same object then update the centroid
            tracking_people_copy=tracking_people.copy()
            current_frame_center_points_copy=current_frame_center_points.copy()
            for p_id, pt_obj in tracking_people_copy.items():

                object_exists=False
                for pt_c in current_frame_center_points_copy:
                    distance= math.hypot(pt_obj[0] -pt_c[0], pt_obj[1] -pt_c[1])
                    # if the Euclidean distance between objects in previous frma 
                    # and current frame is less than 50 then same object
                    if distance<50:
                        # update the center point of the existing object for further tracking
                        tracking_people[p_id]=pt_c
                        object_exists=True
                        current_frame_center_points_copy.remove(pt_c)
                        continue
            # if the object in the dictionary was not found then remove that object from tracking
            if object_exists==False:
                        tracking_people_copy.pop(p_id)
            
            # creating a copy of the trcaking dictionary
            tracking_people_copy=tracking_people.copy()
            
            # if a new an object is found that add the new object
            # for tracking
            for pt_c in current_frame_center_points_copy: 
                 for p_id, pt_obj in tracking_people_copy.items():

                    if pt_c[0] ==pt_obj[0] and pt_c[1] ==pt_obj[1]:
                        continue
                    else:
                        person_id+=1
                        tracking_people[person_id]=pt_c
                        break
        cv2.putText(frame, f'person{person_id}', (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # loop through all the currently trcaked object and 
        # display the trcaked object id
        for p_id, pt_obj in tracking_people.items():
            for pt_c in current_frame_center_points:
                if pt_c[0] ==pt_obj[0] and pt_c[1] ==pt_obj[1]:
                    cv2.putText(frame, f'person{p_id}', (pt_c[0] + 10, pt_c[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                     continue
        cv2.imshow("feed", frame)
        
        # assign the current center points to be used as previous frame center points    
        prev_frame_center_points=current_frame_center_points.copy()
        if cv2.waitKey(1) == ord('q'):
            break
    
video.release()
cv2.destroyAllWindows()
