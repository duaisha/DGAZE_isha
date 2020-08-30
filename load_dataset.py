import os, sys
import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def get_gaze_point(driver, seq, object_annot):
    if seq < 9:
        temp = np.ones((object_annot.shape[0],4))*-1
        return np.hstack((object_annot,temp))
    else:
        bbox = object_annot
        pt1 = (bbox[:,3] + bbox[:,1])/2
        pt2 = (bbox[:,4] + bbox[:,2])/2
        pt1 = np.expand_dims(pt1, axis =1)
        pt2 = np.expand_dims(pt2, axis =1)
        return np.hstack((pt1.astype(int), pt2.astype(int), object_annot[:,1:]))
    
    
def modify_pupil(pupil, face_location):
    
    top_left_x = face_location[:,2:3] 
    top_left_y = face_location[:,0:1]
    pupil[:, [4,6,9]] = pupil[:,[4,6,9]] + top_left_x
    pupil[:, [5,7,10]] = pupil[:,[5,7,10]] + top_left_y
    
    return pupil

def get_driver_features(data_path, driver, sequences, frames_per_seq = None):
    """
    Load driver features for DGAZE:
    Input: Images
    Output: left_eye features of dim = (nframesx36x60x3)
            right_eye features of dim = (nframesx36x60x3)
            # We use face location in driver's input frame for calibration
            face location features of dim = (nframesx4) 
            # face location means driver face location inside cars
            headpose_pupil is of dim = (nframes x 11)
            # (nframes, roll, pitch, yaw, lpupil(x,y), rpupil(x,y), face_area, nose(x,y))
            # x,y left eye pupil location
            gaze_point is ground truth of dim  = (nframesx6)
            # First two values are x,y for point annotation(center of object)
            # Next four are x,y corresponding to top left point of the object and bottom right of the object respectively.
    """

    driver_features = {}
    total_frames = 0
    
    driver_features_path = data_path + '/' + driver
    eye_features_path = driver_features_path + "/explicit_face_features_game/"
    face_features_path = driver_features_path + "/explicit_face_points/"
    annot_path = driver_features_path +"/original_road_view/" 
    
    if os.path.exists(eye_features_path) and os.path.exists(face_features_path) and os.path.exists(annot_path):
        for seq in range(0, sequences):
            features = {}

            eye_features = eye_features_path + "sample" + str(seq+1)
            face_location_features = face_features_path + "sample_" + str(seq+1)
            annot = annot_path + "sample_" + str(seq+1) + ".npy"

            left_eye_features = eye_features + "_left_eye_data.npy"
            right_eye_features = eye_features + "_right_eye_data.npy"
            headpose_pupil_features = eye_features + "_headpose_pupil.npy"
            face_location_features = face_location_features +"_face_points.npy"

            if os.path.exists(left_eye_features) and os.path.exists(right_eye_features) and \
            os.path.exists(headpose_pupil_features) and os.path.exists(face_location_features):

                if np.load(left_eye_features).shape[0] == get_gaze_point(driver, seq, np.load(annot)).shape[0]:

                    features['left_eye']  = np.load(left_eye_features)
                    features['right_eye'] = np.load(right_eye_features)
                    features['headpose_pupil']  = np.load(headpose_pupil_features)
                    features['face_location'] = np.load(face_location_features)
                    features['headpose_pupil'] = modify_pupil(features['headpose_pupil'], features['face_location'])
                    features['gaze_point'] = get_gaze_point(driver, seq, np.load(annot))
                    total_frames += features['gaze_point'].shape[0]

                    driver_features["".join(['seq',str(seq+1)])] = features
                    driver_features['frames_count'] = total_frames
                else:
#                     print(driver, seq+1, np.load(left_eye_features).shape[0], np.load(right_eye_features).shape[0],\
#                           np.load(annot).shape[0], get_gaze_point(driver, seq, np.load(annot)).shape[0])
                    pass
                
    return driver_features


def get_data(data_path, drivers, sequences, frames_per_seq = None):
    data = {}
    for driver in tqdm(drivers):
        data[driver] = get_driver_features(data_path, driver, sequences)
    return data


def load_data(driver_data, driver, sequences, nframes = None):
    gaze_point = None
    left_eye = None
    right_eye = None
    headpose_pupil = None
    face_location = None
    
   
    for ix in tqdm((sequences), position=0, leave=True):
        seq = "".join(['seq',str(ix)]) 
        if seq in driver_data[driver]:
            data = driver_data[driver]["".join(['seq',str(ix)])]

            if gaze_point is None:
                gaze_point = data['gaze_point'][:nframes] if nframes != None else data['gaze_point']
                left_eye = data['left_eye'][:nframes] if nframes != None else data['left_eye']
                right_eye = data['right_eye'][:nframes] if nframes != None else data['right_eye']
                headpose_pupil = data['headpose_pupil'][:nframes] if nframes != None else data['headpose_pupil']
                face_location = data['face_location'][:nframes] if nframes != None else data['face_location']
            else:
                gaze_point = np.concatenate((gaze_point, data['gaze_point'][:nframes]),axis=0)\
                            if nframes != None else np.concatenate((gaze_point, data['gaze_point']),axis=0)
                
                left_eye = np.concatenate((left_eye, data['left_eye'][:nframes]),axis=0)\
                            if nframes != None else np.concatenate((left_eye, data['left_eye']),axis=0)
                
                right_eye = np.concatenate((right_eye, data['right_eye'][:nframes]),axis=0)\
                            if nframes != None else np.concatenate((right_eye, data['right_eye']),axis=0)
                
                headpose_pupil = np.concatenate((headpose_pupil, data['headpose_pupil'][:nframes]),axis=0)\
                            if nframes != None else np.concatenate((headpose_pupil, data['headpose_pupil']),axis=0)
                    
                face_location = np.concatenate((face_location, data['face_location'][:nframes]),axis=0)\
                            if nframes != None else np.concatenate((face_location, data['face_location']),axis=0)
        else:
            #print("Absent --> ",driver, seq)
            pass
    return left_eye, right_eye, headpose_pupil, face_location, gaze_point

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def histogram_equalization(img):
    gray = rgb2gray(img) 
    img_int8 = gray.astype(np.uint8)
    return cv2.equalizeHist(img_int8)

def turker_gaze(data):
    output = np.zeros((data.shape[0], 36,60))
    for i in range(len(data)):
        output[i] = histogram_equalization(data[i])
    return output


def dataset(driver_data, drivers, sequences, nframes = None):
    dgaze_data = {}
    dgaze_left_eye = None
    dgaze_right_eye = None
    dgaze_headpose_pupil = None
    dgaze_face_location = None
    dgaze_gaze_point = None

    for driver in tqdm((drivers), position=0, leave=True):
        left_eye, right_eye, headpose_pupil, face_location, gaze_point = load_data(driver_data, driver, sequences, nframes = nframes)
        
        if dgaze_left_eye is None:
            dgaze_left_eye = left_eye
            dgaze_right_eye = right_eye
            dgaze_headpose_pupil = headpose_pupil
            dgaze_face_location = face_location
            dgaze_gaze_point = gaze_point
        else:
            dgaze_left_eye = np.concatenate((dgaze_left_eye, left_eye),axis=0)
            dgaze_right_eye = np.concatenate((dgaze_right_eye, right_eye),axis=0)
            dgaze_headpose_pupil = np.concatenate((dgaze_headpose_pupil, headpose_pupil),axis=0)
            dgaze_face_location = np.concatenate((dgaze_face_location, face_location),axis=0)
            dgaze_gaze_point = np.concatenate((dgaze_gaze_point, gaze_point),axis=0)
            
     
#     index = (np.where(dgaze_gaze_point[:,1]<1080) and np.where(dgaze_gaze_point[:,1]>=0) and \
#                np.where(dgaze_gaze_point[:,0]<1920) and np.where(dgaze_gaze_point[:,0]>=0))[0]
    
#     print(len(index))
#     print(index)

    x1 = (dgaze_gaze_point[:,0]<1920) & (dgaze_gaze_point[:,0]>=0)
    x2 = (dgaze_gaze_point[:,1]<1080) & (dgaze_gaze_point[:,1]>=0)
    index = x1&x2

    dgaze_data['left_eye'] = (dgaze_left_eye[index]).astype(np.float32)
    dgaze_data['right_eye'] = (dgaze_right_eye[index]).astype(np.float32)
    dgaze_data['headpose_pupil'] = dgaze_headpose_pupil[index].astype(np.float32)
    dgaze_data['face_location'] = dgaze_face_location[index].astype(np.float32)
    dgaze_data['face_features'] = np.concatenate((dgaze_headpose_pupil[:,[1,2,3,4,5,6,7,8,9,10]],dgaze_face_location[:]), axis =1)[index].astype(np.float32)
    dgaze_data['gaze_point'] = dgaze_gaze_point[index].astype(np.float32)
    
    return dgaze_data


