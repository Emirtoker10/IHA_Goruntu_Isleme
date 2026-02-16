# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 17:44:41 2026

@author: Dell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class RenkDedektoru:
    def __init__(self):
        # ROS Ayarları
        rospy.init_node('gazebo_renk_dedektoru', anonymous=True)
        self.bridge = CvBridge()
        
        # ÖNEMLİ: Gazebo kamera topic adını buraya yaz (Örn: /camera/rgb/image_raw)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        
        # Renk Aralıkları (HSV)
        self.blueLower = (90, 50, 50)
        self.blueUpper = (130, 255, 255)
        self.lower_red1, self.upper_red1 = (0, 120, 70), (10, 255, 255)
        self.lower_red2, self.upper_red2 = (170, 120, 70), (180, 255, 255)

    def process_mask(self, mask):
        # Gürültü temizleme (Erozyon ve Genişleme)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def draw_object(self, img, contours, color_name):
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            # Alan eşiği koyalım (çok küçük gürültüleri görmesin)
            if cv2.contourArea(c) > 500:
                rect = cv2.minAreaRect(c)
                box = np.int64(cv2.boxPoints(rect))
                
                # Moment ve Merkez hesaplama
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Çizimler
                    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
                    cv2.circle(img, center, 5, (255, 0, 255), -1)
                    cv2.putText(img, color_name, (center[0]-20, center[1]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def callback(self, data):
        try:
            # ROS Görüntüsünü OpenCV formatına çevir
            imgOriginal = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Hatası: {e}")
            return

        # İşleme Adımları
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Maskeler
        maskBlue = cv2.inRange(hsv, self.blueLower, self.blueUpper)
        maskRed = cv2.bitwise_or(cv2.inRange(hsv, self.lower_red1, self.upper_red1),
                                 cv2.inRange(hsv, self.lower_red2, self.upper_red2))

        # Temizleme
        maskBlue = self.process_mask(maskBlue)
        maskRed = self.process_mask(maskRed)

        # Kontur Bulma
        contoursBlue, _ = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursRed, _ = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Çizim Fonksiyonlarını Çağır
        self.draw_object(imgOriginal, contoursBlue, "MAVI")
        self.draw_object(imgOriginal, contoursRed, "KIRMIZI")

        # Görüntüleme
        cv2.imshow("Gazebo Takip Sistemi", imgOriginal)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        RenkDede = RenkDedektoru()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()