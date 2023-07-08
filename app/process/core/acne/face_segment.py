import os
import cv2
import numpy as np


class Skin_Detect:
    def __init__(self):
        # Constractor that does nothing
        pass

    # RGB bounding rule
    def Rule_A(self, BGR_Frame, plot=False):
        B_Frame, G_Frame, R_Frame = [BGR_Frame[..., BGR] for BGR in range(3)]  # [...] is the same as [:,:]
        # you can use the split built-in method in cv2 library to get the b,g,r components
        # B_Frame, G_Frame, R_Frame  = cv2.split(BGR_Frame)
        # i am using reduce built in method to get the maximum of a 3 given matrices
        BRG_Max = np.maximum.reduce([B_Frame, G_Frame, R_Frame])
        BRG_Min = np.minimum.reduce([B_Frame, G_Frame, R_Frame])
        # at uniform daylight, The skin colour illumination's rule is defined by the following equation :
        Rule_1 = np.logical_and.reduce([R_Frame > 95, G_Frame > 40, B_Frame > 20,
                                        BRG_Max - BRG_Min > 15, abs(R_Frame - G_Frame) > 15,
                                        R_Frame > G_Frame, R_Frame > B_Frame])
        # the skin colour under flashlight or daylight lateral illumination rule is defined by the following equation :
        Rule_2 = np.logical_and.reduce([R_Frame > 220, G_Frame > 210, B_Frame > 170,
                                        abs(R_Frame - G_Frame) <= 15, R_Frame > B_Frame, G_Frame > B_Frame])
        # Rule_1 U Rule_2
        RGB_Rule = np.logical_or(Rule_1, Rule_2)
        # return the RGB mask
        return RGB_Rule

    def lines(self, axis):
        """
        return a list of lines for a give axis
        """
        # equation(3)
        line1 = 1.5862 * axis + 20
        # equation(4)
        line2 = 0.3448 * axis + 76.2069
        # equation(5)
        # the slope of this equation is not correct Cr ≥ -4.5652 × Cb + 234.5652
        # it should be around -1
        line3 = -1.005 * axis + 234.5652
        # equation(6)
        line4 = -1.15 * axis + 301.75
        # equation(7)
        line5 = -2.2857 * axis + 432.85
        return [line1, line2, line3, line4, line5]

    # The five bounding rules of Cr-Cb
    def Rule_B(self, YCrCb_Frame, plot=False):
        Y_Frame, Cr_Frame, Cb_Frame = [YCrCb_Frame[..., YCrCb] for YCrCb in range(3)]
        line1, line2, line3, line4, line5 = self.lines(Cb_Frame)
        YCrCb_Rule = np.logical_and.reduce([line1 - Cr_Frame >= 0,
                                            line2 - Cr_Frame <= 0,
                                            line3 - Cr_Frame <= 0,
                                            line4 - Cr_Frame >= 0,
                                            line5 - Cr_Frame >= 0])
        return YCrCb_Rule

    def Rule_C(self, HSV_Frame, plot=False):
        Hue, Sat, Val = [HSV_Frame[..., i] for i in range(3)]
        # i changed the value of the paper 50 instead of 25 and 150 instead of 230 based on my plots
        HSV_ = np.logical_or(Hue < 50, Hue > 150)
        return HSV_

    def RGB_H_CbCr(self, Frame_, new_img_path):
        Ycbcr_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2YCrCb)
        HSV_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2HSV)
        # Rule A ∩ Rule B ∩ Rule C
        skin_ = np.logical_and.reduce([self.Rule_A(Frame_), self.Rule_B(Ycbcr_Frame), self.Rule_C(HSV_Frame)])

        skin_bw = skin_.astype(np.uint8)
        skin_bw *= 255
        seg = cv2.bitwise_and(Frame_, Frame_, mask=skin_bw)
        # plot as a Grayscale image
        cv2.imwrite(new_img_path, seg)


def face_segmentation(img_path, dir_name, ext_name):
    try:
        img = np.array(cv2.imread(img_path), dtype=np.uint8)
    except:
        print('Error while loading the Image,image does not exist!!!!')

    test = Skin_Detect()
    basename = os.path.basename(img_path)
    basename_without_ext = os.path.splitext(basename)[0]
    new_img_path = os.path.join(dir_name, f"{basename_without_ext}_new{ext_name}")
    test.RGB_H_CbCr(img, new_img_path)

    return new_img_path