import unittest
import os
import math
import sys
import time

import cv2
import numpy as np
sys.path.append('..')

from utils import Panorama


class PanoramaTest(unittest.TestCase):
    def setUp(self):
        self.orig = cv2.imread(os.path.join(os.path.dirname(__file__), 'test_images/original.png'))
        x, y = math.floor(self.orig.shape[0] / 5), math.floor(self.orig.shape[1] / 5)
        self.m = self.orig[x:-x, y:-y, :]

        x, y = math.floor(self.orig.shape[0] / 3), math.floor(self.orig.shape[1] / 3)
        self.tlc =  self.orig[ 0:  x,  0:  y, :]
        self.tm =   self.orig[ x: -x,  0:  y, :]
        self.trc =  self.orig[-x: -1,  0:  y, :]

        self.ml =   self.orig[ 0:  x,  y: -y, :]
        self.mr =   self.orig[-x: -1,  y: -y, :]

        self.llc =  self.orig[ 0:  x, -y: -1, :]
        self.lm =   self.orig[ x: -x, -y: -1, :]
        self.lrc =  self.orig[-x: -1, -y: -1, :]

    def test_image_getter(self):
        pano = Panorama()
        self.assertTrue(np.array_equal(pano.pano, self.m))

    def test_stitcher(self):
        pano = Panorama()


        #cv2.imshow('p0', self.m)
        #print('p0.shape: {}'.format(self.m.shape))
        p, vis = pano.stitch([self.tlc, self.m], showMatches=True)
        p, vis = pano.stitch([self.tm,  p], showMatches=True)
        #cv2.imshow('p2', p)
        #print('p2.shape: {}'.format(p.shape))
        #p, vis = pano.stitch([self.trc, p], showMatches=True)
        #p, vis = pano.stitch([self.ml, p], showMatches=True)
        #p, vis = pano.stitch([self.ml, p], showMatches=True)
        #p, vis = pano.stitch([self.llc, p], showMatches=True)
        #p, vis = pano.stitch([self.lm,  p], showMatches=True)
        #p, vis = pano.stitch([self.lrc, p], showMatches=True)


        #self.assertEqual(self.orig.shape, pano.pano.shape)
        #self.assertTrue(np.array_equal(pano.pano, self.m))
        #cv2.imshow('A', self.m)
        #cv2.imshow('B', self.tlc)
        #cv2.imshow('vis', vis)
        cv2.imshow('pano 1', p)
        cv2.waitKey(15000)
    

if __name__ == "__main__":
    unittest.main()
