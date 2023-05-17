import numpy
import cv2
import glob


category = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

points_of_object = numpy.zeros((6*9,3), numpy.float32)
points_of_object[:,:2] = numpy.mgrid[0:9,0:6].T.reshape(-1,2)
points_of_object = 21.5*points_of_object
#empty lists to store image and object points
object_list = []
image_list = []


#To read all the images from the folder
Calibrate_img = glob.glob('./Calibration_Imgs/*.jpg')
for fname in Calibrate_img:
    obj_list = []
    img_list = []
    pic = cv2.imread(fname)
    #resizing and convertion to grayscale

    grayscale_mask = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    #finding the corners
    seg, corners = cv2.findChessboardCorners(grayscale_mask, (9,6), None)

    if seg == True:
        object_list.append(points_of_object)
        obj_list.append(points_of_object)
        corner_fn = cv2.cornerSubPix(grayscale_mask,corners, (11,11), (-1,-1), category)
        image_list.append(corner_fn)
        img_list.append(corner_fn)

        cv2.drawChessboardCorners(pic, (9,6), corner_fn, seg)

        cv2.imshow('pic', pic)
        cv2.waitKey(1000)
        inaccuracy_mean = 0
        inaccuracy_mean_ = 0
        #calculating K matrix, rotation, translation and distortion using calibrate camera function
        seg, k_mat, distortion, rot, translation = cv2.calibrateCamera(object_list, image_list, grayscale_mask.shape[::-1], None, None)
        seg_, k_mat_, distortion_, rot_, translation_ = cv2.calibrateCamera(obj_list, img_list, grayscale_mask.shape[::-1], None, None)

        for i in range(len(object_list)):
             points_of_img, _ = cv2.projectPoints(object_list[i], rot[i], translation[i], k_mat, distortion)
             inaccuracy = cv2.norm(image_list[i], points_of_img, cv2.NORM_L2)/len(points_of_img)
             inaccuracy_mean += inaccuracy
        print("K matrix is: \n", k_mat)
        print( "reprojection error till image: {}".format(inaccuracy_mean/len(object_list)) )

        for i in range(len(obj_list)):
             points_of_img_, _ = cv2.projectPoints(obj_list[i], rot_[i], translation_[i], k_mat_, distortion_)
             inaccuracy_ = cv2.norm(img_list[i], points_of_img_, cv2.NORM_L2)/len(points_of_img_)
             inaccuracy_mean_ += inaccuracy_
        print( "reprojection error for image: {}".format(inaccuracy_mean_/len(obj_list)) )
             
cv2.destroyAllWindows()