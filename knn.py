import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

base_dir_cars = 'cars/'
base_dir_noncars = 'noncars/'
base_dir_test = 'cars_test/'

img_filenames_cars = os.listdir(base_dir_cars)
img_filenames_noncars = os.listdir(base_dir_noncars)

img_kps_surf = list()
img_kps_orb = list()

img_desc_surf = list()
img_desc_orb = list()

img_labels = list()

surf = cv2.SURF(5000)
orb = cv2.ORB()

for i in xrange(len(img_filenames_cars)):
  full_filename = base_dir_cars + img_filenames_cars[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  img_kps_surf.append(surf_kp)
  img_desc_surf.append(surf_desc)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  img_kps_orb.append(orb_kp)
  img_desc_orb.append(orb_desc)
  img_labels.append(0)

for i in xrange(len(img_filenames_noncars)):
  full_filename = base_dir_noncars + img_filenames_noncars[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  img_kps_surf.append(surf_kp)
  img_desc_surf.append(surf_desc)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  img_kps_orb.append(orb_kp)
  img_desc_orb.append(orb_desc)
  img_labels.append(1)

knn_surf = cv2.KNearest()
knn_surf.train(img_kps_surf, img_labels)

knn_orb = cv2.KNearest()
knn_orb.train(img_kps_orb, img_labels)

img_filenames_test = os.listdir(base_dir_test)

for i in xrange(len(img_filenames_test)):
  full_filename = base_dir_test + img_filenames_test[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  surf_ret, surf_result, surf_neighbours, surf_dist = knn_surf.find_nearest(surf_kp, k = 5)
  print "Surf Result: " + str(surf_result)
  print "Surf Neighbors: " + str(surf_neighbours)
  print "Surf Distance: " + str(surf_dist)
  orb_ret, orb_result, org_neighbours, orb_dist = knn_orb.find_nearest(org_kp, k = 5)
  print "Orb Result: " + str(orb_result)
  print "Orb Neighbors: " + str(org_neighbours)
  print "Orb Distance: " + str(orb_dist)

print "Done"