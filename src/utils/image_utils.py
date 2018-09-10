import numpy as np
import imutils
import cv2
import math
import sys
import time


class Panorama:
    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
        showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        cv2.imshow('A1', imageA)
        cv2.imshow('B1', imageB)
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
         featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None   

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, T, status) = M
        T[0:2, 0:2] = np.array(((1, 0), (0, 1)))
        x_trans = int(T[1][2]) 
        y_trans = int(T[0][2]) 
        print('x_trans: {}'.format(x_trans))
        print('y_trans: {}'.format(y_trans))

        if x_trans < 0 and y_trans < 0:
            bx_start  = abs(x_trans)
            by_start  = abs(y_trans)
            imageA = np.pad(imageA, ((0, abs(imageB.shape[0] + abs(x_trans) - imageA.shape[0])), (0, abs(imageB.shape[1] + abs(y_trans) - imageA.shape[1])), (0, 0)), 'constant')
            #imageA = np.pad(imageA, ((0, imageB.shape[0] + abs(x_trans)), (0, imageB.shape[1] + abs(y_trans)), (0, 0)), 'constant')
        elif x_trans < 0 and y_trans >= 0:
            bx_start  = abs(x_trans)
            by_start  = 0
            imageA = np.pad(imageA, ((0, abs(imageB.shape[0] + abs(x_trans) - imageA.shape[0])), (y_trans, 0), (0, 0)), 'constant')
        elif x_trans >= 0 and y_trans < 0:
            bx_start  = 0
            by_start  = abs(y_trans)
            imageA = np.pad(imageA, ((x_trans, 0), (0, abs(imageB.shape[1] + abs(y_trans) - imageA.shape[1])), (0, 0)), 'constant')
        elif x_trans >= 0 and y_trans >= 0:
            bx_start  = 0
            by_start  = 0
            imageA = np.pad(imageA, ((x_trans, 0), (y_trans, 0), (0, 0)), 'constant')

        result = imageA
        #result = cv2.warpAffine(imageA, T,
        #    (imageA.shape[1], imageA.shape[0]))
        result[bx_start:bx_start + imageB.shape[0], by_start:by_start + imageB.shape[1]] = imageB
        #cv2.imshow('imageB', imageB)
        #cv2.imshow('result', result)

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SURF_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
 
        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
 
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
 
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
 
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append((m.trainIdx, m.queryIdx))
                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            T = cv2.estimateRigidTransform(ptsA, ptsB, False)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, T, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis



class RegionGraph:
    def __init__(self, image, indirect_neighbors=True):
        '''
        image: an image in numpy BGR format
        indirect_neighbors: Boolean value
        ''' 
        self.image = image

    def generate_graph(self):
        pass

    def _get_neighbors(self, x, y, indirect_neighbors=True):
        '''
        x, y: coordinates of central point
        indirect_neighbors:
            False: Do NOT include diagonals when getting neighbors. Returns a max of four values.
                  nyn
                  y.y
                  nyn
            True: Include diagonals when getting neighbors. Returns a max of eight values.
                  yyy
                  y.y
                  yyy
        '''
        pass

