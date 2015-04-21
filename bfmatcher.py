import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

base_dir_train = 'cars_train/'
base_dir_test = 'cars_test/'

img_kps_surf = list()
img_kps_orb = list()
img_desc_surf = list()
img_desc_orb = list()
img_filenames_train = os.listdir(base_dir_train)
img_matches = list()

surf = cv2.SURF(5000)
orb = cv2.ORB()
bf_norm_l2 = cv2.BFMatcher(cv2.NORM_L2)
bf_norm_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)

for i in xrange(len(img_filenames_train)):
  full_filename = base_dir_train + img_filenames_train[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  img_kps_surf.append(surf_kp)
  img_desc_surf.append(surf_desc)
  orb_kp = orb.detect(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  img_kps_orb.append(orb_kp)
  img_desc_orb.append(orb_desc)

img_filenames_test = os.listdir(base_dir_test)

for i in xrange(len(img_filenames_test)):
  has_car_surf = False
  has_car_orb = False
  full_filename = base_dir_test + img_filenames_test[i]
  print full_filename
  img = cv2.imread(full_filename, 0)
  surf_kp, surf_desc = surf.detectAndCompute(img, None)
  orb_kp, orb_desc = orb.compute(img, orb_kp)
  matches_surf = bf_norm_l2.match(img_desc_surf[i], surf_desc)
  print "Matches SURF: " + str(len(matches_surf))
  matches_orb = bf_norm_l2.match(img_desc_orb[i], orb_desc)
  print "Matches ORB: " + str(len(matches_orb))
  if len(matches_surf) > 100:
    has_car_surf = True
  if len(matches_orb) > 350:
    has_car_orb = True
  print full_filename + " has car? Surf: " + str(has_car_surf) + " Orb: " + str(has_car_orb)

print "Done"
