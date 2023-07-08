import os
import cv2
import dlib
from imutils import face_utils


def crop_boundary(top, bottom, left, right, faces):
    if faces:
        top = max(0, top - 120)
        left = max(0, left - 50)
        right += 30
        bottom += 30
    else:
        top = max(0, top - 50)
        left = max(0, left - 50)
        right += 50
        bottom += 50

    return top, bottom, left, right


def crop_face(img_path):
    try:
        frame = cv2.imread(img_path)
        # basename = os.path.basename(img_path)
        # basename_without_ext = os.path.splitext(basename)[0]
        if frame is None:
            return print(f"Invalid file path: [{img_path}]"), False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(gray, 1)
        if not len(rects):
            return print(f"Sorry. HOG could not detect any faces from your image.\n[{img_path}]"), False

        crop_img_path = img_path
        # for (i, rect) in enumerate(rects):
        #     (x, y, w, h) = face_utils.rect_to_bb(rect)
        #
        #     top, bottom, left, right = crop_boundary(y, y + h, x, x + w, len(rects) <= 2)
        #     crop_img_path = os.path.join(dir_name, f"{basename_without_ext}_crop{ext_name}")
        #     crop_img = frame[top: bottom, left: right]
        #     cv2.imwrite(crop_img_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        # print(f"SUCCESS: [{basename}]")

        return crop_img_path, True

    except Exception as e:
        return print('face detect have error--->', e), False
