import cv2
import os
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt

base_dir_cars = 'cars_train/'
base_dir_noncars = 'noncars_train/'
base_dir_cars_test = 'cars_test/'
base_dir_noncars_test = 'noncars_test/'

img_filenames_cars = os.listdir(base_dir_cars)
img_filenames_noncars = os.listdir(base_dir_noncars)
img_filenames_cars_test = os.listdir(base_dir_cars_test)
img_filenames_noncars_test = os.listdir(base_dir_noncars_test)

img_desc_surf = list()
img_desc_orb = list()

img_desc_surf_labels = list()
img_desc_orb_labels = list()

total_correct_surf = 0
total_type_one_error_surf = 0
total_type_two_error_surf = 0

total_correct_orb = 0
total_type_one_error_orb = 0
total_type_two_error_orb = 0

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
  if surf_desc is not None:
    for j in xrange(len(surf_desc)):
      img_desc_surf.append(surf_desc[j])
      img_desc_surf_labels.append(0)
  # Get features using ORB
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  # Append features to ORB features list
  # And add labels for ORB features
  if orb_desc is not None:
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
  if surf_desc is not None:
    for j in xrange(len(surf_desc)):
      img_desc_surf.append(surf_desc[j])
      img_desc_surf_labels.append(1)
  # Get features using ORB
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  # Append features to ORB features list
  # And add labels for ORB features
  if orb_desc is not None:
    for j in xrange(len(orb_desc)):
      img_desc_orb.append(orb_desc[j])
      img_desc_orb_labels.append(1)

print "Training with SURF data..."
surf_clf = svm.SVC(kernel='rbf', gamma=0.001)
surf_clf.fit(img_desc_surf, img_desc_surf_labels)

print "Training with ORB data..."
orb_clf = svm.SVC(gamma=0.001, C=100.)
orb_clf.fit(img_desc_orb, img_desc_orb_labels)

print "Starting classification..."

for i in xrange(len(img_filenames_noncars_test)):
  full_filename = base_dir_noncars_test + img_filenames_noncars_test[i]
  print "Predicting " + full_filename + " ..."
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  surf_prediction = surf_clf.predict(surf_desc)
  orb_prediction = orb_clf.predict(orb_desc)
  if surf_prediction[0] == 1:
    total_correct_surf = total_correct_surf + 1
  else:
   total_type_one_error_surf = total_type_one_error_surf + 1
  if orb_prediction[0] == 1:
    total_correct_orb = total_correct_orb + 1
  else:
    total_type_one_error_orb = total_type_one_error_orb + 1

for i in xrange(len(img_filenames_cars_test)):
  full_filename = base_dir_cars_test + img_filenames_cars_test[i]
  print "Predicting " + full_filename + " ..."
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  surf_prediction = surf_clf.predict(surf_desc)
  orb_prediction = orb_clf.predict(orb_desc)
  if surf_prediction[0] == 1:
    total_correct_surf = total_correct_surf + 1
  else:
   total_type_one_error_surf = total_type_one_error_surf + 1
  if orb_prediction[0] == 1:
    total_correct_orb = total_correct_orb + 1
  else:
    total_type_one_error_orb = total_type_one_error_orb + 1

print "Total Correct Surf: " + str(total_correct_surf)
print "Total Type 1 Error Surf: " + str(total_type_one_error_surf)
print "Total Type 2 Error Surf: " + str(total_type_two_error_surf)
print "Total Correct Orb: " + str(total_correct_orb)
print "Total Type 1 Error Orb: " + str(total_type_one_error_orb)
print "Total Type 2 Error Orb: " + str(total_type_two_error_orb)
print "Total Number Training Images: " + str(len(img_filenames_noncars) + len(img_filenames_cars))
print "Total Number Test Images: " + str(len(img_filenames_noncars_test) + len(img_filenames_cars_test))