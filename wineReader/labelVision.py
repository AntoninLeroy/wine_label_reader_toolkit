import cv2
import numpy as np
import imutils
from keras.preprocessing.image import save_img
import pytesseract
import PIL.Image
from math import ceil

# Copyright notice: a part of the code is pulled from https://github.com/Nepherhotep/unwrap_labels with the author autorisation.

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)

class Line(object):
    def __init__(self, point1, point2):
        """
        For line formula y(x) = k * x + b, calc k and b params
        If the line is vertical, set "vertical" attr to True and save "x" position of the line
        """
        self.point1 = point1
        self.point2 = point2
        self.vertical = False
        self.fixed_x = None
        self.k = None
        self.b = None

        # cached angle props
        self.angle = None
        self.angle_cos = None
        self.angle_sin = None

        self.set_line_props(point1, point2)

    def is_vertical(self):
        return self.vertical

    def set_line_props(self, point1, point2):
        if point2[0] - point1[0]:
            self.k = float(point2[1] - point1[1]) / (point2[0] - point1[0])
            self.b = point2[1] - self.k * point2[0]

            k_normal = - 1 / self.k
        else:
            self.vertical = True
            self.fixed_x = point2[0]

            k_normal = 0

        self.angle = np.arctan(k_normal)
        self.angle_cos = np.cos(self.angle)
        self.angle_sin = np.sin(self.angle)

    def get_x(self, y):
        if self.is_vertical():
            return self.fixed_x
        else:
            return int(round(float(y - self.b) / self.k))

    def get_y(self, x):
        return self.k * x + self.b

class LabelUnwrapper(object):
    COL_COUNT = 30
    ROW_COUNT = 20

    def __init__(self, src_image=None, pixel_points=None, percent_points=None):
        """
        Point lists are lists of 6 points - [A, B, C, D, E, F]
        :param pixel_points: List[Tuple] Points, whose coordinates specified as pixels
        :param percent_points: List[Tuple] Points, whose coordinates specified as fraction of image width/height
        In both cases points represent figure below:
        |        |                  |        |
        |    B   |                  A        C
        | /    \ |                  | \    / |
        A        C                  |   B    |
        |        |                  |        |
        |        |       OR         |        |
        |        |                  |        |
        F        D                  F        D
        | \    / |                  | \    / |
        |   E    |                  |   E    |
        |        |                  |        |
        So, A-B-C-D-E-F-A polygon represent raw wine label on bottle
        """
        self.src_image = src_image
        self.width = self.src_image.shape[1]
        self.height = src_image.shape[0]

        self.dst_image = None
        self.points = pixel_points
        self.percent_points = percent_points

        self.point_a = None  # top left
        self.point_b = None  # top center
        self.point_c = None  # top right
        self.point_d = None  # bottom right
        self.point_e = None  # bottom center
        self.point_f = None  # bottom left

        self.center_line = None
        self.load_points()

    def load_points(self):
        if self.points is None:
            points = []
            for point in self.percent_points:
                x = int(point[0] * self.width)
                y = int(point[1] * self.height)
                points.append((x, y))

            self.points = points

        self.points = np.array(self.points)
        (self.point_a, self.point_b, self.point_c,
         self.point_d, self.point_e, self.point_f) = self.points

        center_top = (self.point_a + self.point_c) / 2
        center_bottom = (self.point_d + self.point_f) / 2

        self.center_line = Line(center_bottom, center_top)
        if not len(self.points) == 6:
            raise ValueError("Points should be an array of 6 elements")

    def unwrap(self, interpolate=False):
        source_map = self.calc_source_map()
        if interpolate:
            self.unwrap_label_interpolation(source_map)
        else:
            self.unwrap_label_perspective(source_map)
        return self.dst_image

    def calc_dest_map(self):
        width, height = self.get_label_size()

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                row.append([int(dx * col_index),
                            int(dy * row_index)])

            rows.append(row)
        return np.array(rows)

    def unwrap_label_interpolation(self, source_map):
        """
        Unwrap label using interpolation - more accurate method in terms of quality
        """
        from scipy.interpolate import griddata

        width, height = self.get_label_size()

        dest_map = self.calc_dest_map()

        grid_x, grid_y = np.mgrid[0:width - 1:width * 1j, 0:height - 1:height * 1j]

        destination = dest_map.reshape(dest_map.size // 2, 2)
        source = source_map.reshape(source_map.size // 2, 2)

        grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
        map_x = np.append([], [ar[:, 0] for ar in grid_z]).reshape(width, height)
        map_y = np.append([], [ar[:, 1] for ar in grid_z]).reshape(width, height)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        warped = cv2.remap(self.src_image, map_x_32, map_y_32, cv2.INTER_CUBIC)
        self.dst_image = cv2.transpose(warped)

    def unwrap_label_perspective(self, source_map):
        """
        Unwrap label using transform, unlike unwrap_label_interpolation doesn't require scipy
        """
        width, height = self.get_label_size()
        self.dst_image = np.zeros((height, width, 3), np.uint8)

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        dx_int = int(np.ceil(dx))
        dy_int = int(np.ceil(dy))

        for row_index in range(self.ROW_COUNT - 1):
            for col_index in range(self.COL_COUNT - 1):
                src_cell = (source_map[row_index][col_index],
                            source_map[row_index][col_index + 1],
                            source_map[row_index + 1][col_index],
                            source_map[row_index + 1][col_index + 1])

                dst_cell = np.int32([[0, 0], [dx, 0], [0, dy], [dx, dy]])

                M = cv2.getPerspectiveTransform(np.float32(src_cell), np.float32(dst_cell))
                dst = cv2.warpPerspective(self.src_image, M, (dx_int, dy_int))
                x_offset = int(dx * col_index)
                y_offset = int(dy * row_index)

                self.dst_image[y_offset:y_offset + dy_int,
                               x_offset:x_offset + dx_int] = dst

    def get_roi_rect(self, points):
        max_x = min_x = points[0][0]
        max_y = min_y = points[0][1]
        for point in points:
            x, y = point
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def get_roi(self, image, points):
        rect = self.get_roi_rect(points)
        return image[np.floor(rect[0][1]):np.ceil(rect[2][1]),
                     np.floor(rect[0][0]):np.ceil(rect[1][0])]

    def calc_source_map(self):
        top_points = self.calc_ellipse_points(self.point_a, self.point_b, self.point_c,
                                              self.COL_COUNT)
        bottom_points = self.calc_ellipse_points(self.point_f, self.point_e, self.point_d,
                                                 self.COL_COUNT)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                top_point = top_points[col_index]
                bottom_point = bottom_points[col_index]

                delta = (top_point - bottom_point) / float(self.ROW_COUNT - 1)

                point = top_point - delta * row_index
                row.append(point)
            rows.append(row)
        return np.array(rows)

    def draw_mesh(self, color=YELLOW_COLOR, thickness=3):
        mesh = self.calc_source_map()
        for row in mesh:
            for x, y in row:
                point = (int(round(x)), int(round(y)))
                cv2.line(self.src_image, point, point, color=color, thickness=ceil(self.width*0.01))

    def draw_poly_mask(self, color=WHITE_COLOR):
        cv2.polylines(self.src_image, np.int32([self.points]), 1, color)

    def draw_mask(self, color=WHITE_COLOR, thickness=1, img=None):
        """
        Draw mask, if image not specified - draw to source image
        """
        if img is None:
            img = self.src_image

        cv2.line(img, tuple(self.point_f.tolist()), tuple(self.point_a.tolist()), color, thickness)
        cv2.line(img, tuple(self.point_c.tolist()), tuple(self.point_d.tolist()), color, thickness)

        self.draw_ellipse(img, self.point_a, self.point_b, self.point_c, color, thickness)
        self.draw_ellipse(img, self.point_d, self.point_e, self.point_f, color, thickness)

    def get_label_contour(self, color=WHITE_COLOR, thickness=1):
        mask = np.zeros(self.src_image.shape)
        self.draw_mask(color, thickness, mask)
        return mask

    def get_label_mask(self):
        """
        Generate mask of the label, fully covering it
        """
        mask = np.zeros(self.src_image.shape)
        pts = np.array([[self.point_a, self.point_c, self.point_d, self.point_f]])
        cv2.fillPoly(mask, pts, WHITE_COLOR)
        self.draw_filled_ellipse(mask, self.point_a, self.point_b, self.point_c, True)
        self.draw_filled_ellipse(mask, self.point_f, self.point_e, self.point_d, False)
        return mask

    def draw_ellipse(self, img, left, top, right, color=WHITE_COLOR, thickness=1):
        """
        Draw ellipse using opencv function
        """
        is_arc, center_point, axis, angle = self.get_ellipse_params(left, top, right)

        if is_arc:
            start_angle, end_angle = 0, 180
        else:
            start_angle, end_angle = 180, 360

        cv2.ellipse(img, center_point, axis, angle, start_angle, end_angle, color, thickness)

    def draw_filled_ellipse(self, img, left, top, right, is_top=False):
        is_arc, center_point, axis, angle = self.get_ellipse_params(left, top, right)

        if is_arc ^ is_top:
            color = WHITE_COLOR
        else:
            color = BLACK_COLOR

        cv2.ellipse(img, center_point, axis, angle, 0, 360, color=color, thickness=-1)

    def get_ellipse_params(self, left, top, right):
        center = (left + right) / 2
        center_point = tuple(map(lambda x: int(np.round(x)), center.tolist()))

        axis = (int(np.linalg.norm(left - right) / 2), int(np.linalg.norm(center - top)))

        x, y = left - right
        angle = np.arctan(float(y) / x) * 57.296

        is_arc = False
        if (top - center)[1] > 0:
            is_arc = True

        return is_arc, center_point, axis, angle

    def calc_ellipse_points(self, left, top, right, points_count):
        center = (left + right) / 2

        # get ellipse axis
        a = np.linalg.norm(left - right) / 2
        b = np.linalg.norm(center - top)

        # get start and end angles
        if (top - center)[1] > 0:
            delta = np.pi / (points_count - 1)

        else:
            delta = - np.pi / (points_count - 1)

        cos_rot = (right - center)[0] / a
        sin_rot = (right - center)[1] / a

        points = []
        for i in range(points_count):
            phi = i * delta
            dx, dy = self.get_ellipse_point(a, b, phi)

            x = round(center[0] + dx * cos_rot - dy * sin_rot)
            y = round(center[1] + dx * sin_rot + dy * cos_rot)

            points.append([x, y])

        points.reverse()
        return np.array(points)

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)

    def get_label_size(self):
        top_left = self.point_a
        top_right = self.point_c
        bottom_right = self.point_d
        bottom_left = self.point_f

        width1 = np.linalg.norm(top_left - top_right)
        width2 = np.linalg.norm(bottom_left - bottom_right)
        avg_width = int((width1 + width2) * np.pi / 4)

        height1 = np.linalg.norm(top_left - bottom_left)
        height2 = np.linalg.norm(top_right - bottom_right)
        avg_height = int((height1 + height2) / 2)
        return avg_width, avg_height

class labelVision:

    def __init__(self, Config):
        self.Config = Config

    def align_vertically(self, src, mask):
            
            # img must be grayscale 2D numpy array
            img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            degree = []
            to_max = []
            step = 1
            init_black_col = img.shape[1] - np.count_nonzero(img.sum(axis = 0))
            
            # rotating positive or negative ?
            rot_pos_black_col = img.shape[1] - np.count_nonzero(imutils.rotate(img, 1).sum(axis = 0))
            
            if rot_pos_black_col > init_black_col:
                
                # positive rotation
                for deg in range(0,45,step):
                    black_col = img.shape[1] - np.count_nonzero(imutils.rotate(img, deg).sum(axis = 0))
                    degree.append(deg)
                    to_max.append(black_col)
                    
            else:
                
                # negative rotation
                for deg in range(0,-45,-step):
                    black_col = img.shape[1] - np.count_nonzero(imutils.rotate(img, deg).sum(axis = 0))
                    degree.append(deg)
                    to_max.append(black_col)
            
            max_value = max(to_max)
            max_index = to_max.index(max_value)
            rot_degree = degree[max_index]
            
            mask = imutils.rotate(mask, rot_degree)
            src = imutils.rotate(src, rot_degree)
            
            return src, mask

    def getCylinderPoints(self, mask):

        # Const
        x_length = mask.shape[1]
        y_length = mask.shape[0]

        # Loolink for F point via diagonal
        # approach from botom left to top right diagonally

        for off in reversed(range(y_length)):
            found = False
            for diag in np.diagonal(mask, offset=-off, axis1=0, axis2=1):
                if diag.sum() > 0:
                    found = True
                    break
            if found:
                break
            
        for i, e in enumerate(diag):
            if e > 0:
                Xf = i
                Yf = int(y_length - (diag.shape[0] - i))
                break
                
        # Loolink for C point via diagonal
        # approach from top right to bottom left diagonally

        for off in reversed(range(x_length)):
            found = False
            for diag in np.diagonal(mask, offset=off, axis1=0, axis2=1):
                if diag.sum() > 0:
                    found = True
                    break
            if found:
                break
            
        for i, e in enumerate(diag):
            if e > 0:
                Yc = i
                Xc = int(x_length - (diag.shape[0] - i))
                break
                
        # Loolink for D point via diagonal
        # approach from botom right to top left we need to use inverse diagonal
        flip = np.fliplr(mask)

        for off in reversed(range(y_length)):
            found = False
            for diag in np.diagonal(flip, offset=-off, axis1=0, axis2=1):
                if diag.sum() > 0:
                    found = True
                    break
            if found:
                break
            
        for i, e in enumerate(diag):
            if e > 0:
                Xd = x_length - i
                Yd = int(y_length - (diag.shape[0] - i))
                break
                
        # Loolink for A point via diagonal
        # approach from top right to bottom left we need to use inverse diagonal

        for off in reversed(range(x_length)):
            found = False
            for diag in np.diagonal(flip, offset=off, axis1=0, axis2=1):
                if diag.sum() > 0:
                    found = True
                    break
            if found:
                break
            
        for i, e in enumerate(diag):
            if e > 0:
                Ya = i
                Xa = int(diag.shape[0] - i)
                break
                
        # Looking for B point by simple scaning row from top to bottom in the midle of A and C point until first pixel

        Xb = int((Xa + Xc)/2)

        for i, e in enumerate(mask[:,Xb]):
            if e.sum() > 0:
                Yb = i
                break

        # Looking for E point by simple scaning row from bottom to top in the midle of D and F point until first pixel

        Xe = int((Xf + Xd)/2)

        for i in reversed(range(y_length)):
            if mask[i,Xe].sum() > 0:
                Ye = i
                break

        mask = cv2.circle(mask, (Xf, Yf), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)
        mask = cv2.circle(mask, (Xc, Yc), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)
        mask = cv2.circle(mask, (Xd, Yd), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)
        mask = cv2.circle(mask, (Xa, Ya), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)
        mask = cv2.circle(mask, (Xb, Yb), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)
        mask = cv2.circle(mask, (Xe, Ye), radius=ceil(x_length*0.02), color=(255, 0, 0), thickness=-1)

        # convert points to ratio of height and width

        Xa = Xa / x_length
        Xb = Xb / x_length
        Xc = Xc / x_length
        Xd = Xd / x_length
        Xe = Xe / x_length
        Xf = Xf / x_length

        Ya = Ya / y_length
        Yb = Yb / y_length
        Yc = Yc / y_length
        Yd = Yd / y_length
        Ye = Ye / y_length
        Yf = Yf / y_length

        shape = {"tag": "label", "shape": [{"x": Xa, "y": Ya}, #A
                                   {"x": Xb, "y": Yb}, #B
                                   {"x": Xc, "y": Yc}, #C
                                   {"x": Xd, "y": Yd}, #D
                                   {"x": Xe, "y": Ye}, #E
                                   {"x": Xf, "y": Yf}]} #F

        return shape, mask

    def unwrapLabel(self, src, shape):

        points = []
        for point in shape['shape']:
            points.append([point['x'], point['y']])

        unwrapper = LabelUnwrapper(src_image=src, percent_points=points)

        dst_image = unwrapper.unwrap()
        for point in unwrapper.points:
            cv2.line(unwrapper.src_image, tuple(point), tuple(point), color=RED_COLOR, thickness=ceil(src.shape[1]*0.04))

        unwrapper.draw_mesh()

        return src, dst_image

    def ocr(self, src, unwrapped):
        # Optical character recognition
        src_ocr = pytesseract.image_to_string(src, config = self.Config['ocr_mode'])
        unwrapped_ocr = pytesseract.image_to_string(unwrapped, config = self.Config['ocr_mode'])
        
        return unwrapped_ocr, src_ocr

    def readLabels(self, unet_output, srcs, fileNames):

        for mask, src, filename in zip(unet_output, srcs, fileNames):

            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
            mask=cv2.resize(mask,(src.shape[1],src.shape[0]))
            mask = np.round(mask) #binary transform

            save_img(self.Config['results_path'] + filename + "/3_resize_binary_mask.jpg", mask)

            # rotate
            r_src, r_mask = self.align_vertically(src, mask)
            save_img(self.Config['results_path'] + filename + "/4_rotate_mask.jpg", r_mask)
            save_img(self.Config['results_path'] + filename + "/5_rotate_src.jpg", r_src)

            # search cylindric edges points in label structure
            r_mask = np.uint8(r_mask*255)
            shape, img_point = self.getCylinderPoints(r_mask)
            save_img(self.Config['results_path'] + filename + "/6_points.jpg", img_point)

            # unwrap label
            mesh, unwrapped = self.unwrapLabel(r_src, shape)
            cv2.imwrite(self.Config['results_path'] + filename + "/7_mesh.jpg", mesh)
            cv2.imwrite(self.Config['results_path'] + filename + "/8_unwrapped.jpg", unwrapped)

            # tesseract ocr
            unwrapped_ocr, src_ocr = self.ocr(src, unwrapped)
            to_write_unw_ocr=open(self.Config['results_path'] + filename + "/9_unwrapped_ocr.txt",'w')
            to_write_unw_ocr.write(unwrapped_ocr)
            to_write_unw_ocr.close()
            to_write_src_ocr=open(self.Config['results_path'] + filename + "/10_source_ocr.txt",'w')
            to_write_src_ocr.write(src_ocr)
            to_write_src_ocr.close()