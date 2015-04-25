import cv2
import os
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt

base_dir_cars = 'cars/'
base_dir_noncars = 'noncars/'
base_dir_test = 'cars_test/'

img_filenames_cars = os.listdir(base_dir_cars)
img_filenames_noncars = os.listdir(base_dir_noncars)
img_filenames_test = os.listdir(base_dir_test)

img_desc_surf = list()
img_desc_orb = list()

img_desc_surf_labels = list()
img_desc_orb_labels = list()

surf = cv2.SURF(5000)
orb = cv2.ORB()

# Get training data for known cars
print "Extracting features..."
for i in xrange(len(img_filenames_cars)):
  full_filename = base_dir_cars + img_filenames_cars[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  # Get features using SURF
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  # Append features to SURF features list
  # And add labels for SURF features
  for j in xrange(len(surf_desc)):
    img_desc_surf.append(surf_desc[j])
    img_desc_surf_labels.append(0)
  # Get features using ORB
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  # Append features to ORB features list
  # And add labels for ORB features
  for j in xrange(len(orb_desc)):
    img_desc_orb.append(orb_desc[j])
    img_desc_orb_labels.append(0)

# Get training data for known non cars
for i in xrange(len(img_filenames_noncars)):
  full_filename = base_dir_noncars + img_filenames_noncars[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  # Get features using SURF
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  # Append features to SURF features list
  # And add labels for SURF features
  for j in xrange(len(surf_desc)):
    img_desc_surf.append(surf_desc[j])
    img_desc_surf_labels.append(1)
  # Get features using ORB
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  # Append features to ORB features list
  # And add labels for ORB features
  for j in xrange(len(orb_desc)):
    img_desc_orb.append(orb_desc[j])
    img_desc_orb_labels.append(1)

print "Training with SURF data..."
surf_clf = svm.SVC(gamma=0.001, C=100.)
surf_clf.fit(img_desc_surf, img_desc_surf_labels)

print "Training with ORB data..."
orb_clf = svm.SVC(gamma=0.001, C=100.)
orb_clf.fit(img_desc_orb, img_desc_orb_labels)

print "Starting classification..."

for i in xrange(len(img_filenames_test)):
  full_filename = base_dir_test + img_filenames_test[i]
  print "Predicting " + full_filename + " ..."
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  surf_prediction = surf_clf.predict(surf_desc)
  orb_prediction = orb_clf.predict(orb_desc)
  print "Surf Prediction: " + str(surf_prediction[0])
  print "ORB Prediction: " + str(orb_prediction[0])