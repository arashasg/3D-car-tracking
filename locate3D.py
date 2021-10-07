import cv2
import numpy as np
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt

# Prepare the pane
img = np.ones((375, 1242, 3), dtype=np.uint8)
img = img * 255


class Utils:

    def __init__(self):
        x_p90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        y_m90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.cord_align = np.dot(y_m90, x_p90)
        self.solvers = [self.__solve_front, self.__solve_right, self.__solve_rear,
                        self.__solve_left, self.__solve_roof]
        self.corners3D = None
        self.corners2D = None
        self.finalR = np.eye(3)

    def rotate(self, yaw):
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = np.dot(self.cord_align, R)
        return R

    def __solve_front(self, pix_pos2D, R, dims, center, cam_to_img):

        # Prepare variables
        X = dims[0] / 2  # dx/2
        Y = None
        Z = None

        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]

        R1 = np.zeros(shape=(3, 2))
        R1[:, 0] = r2
        R1[:, 1] = r3

        b = np.reshape(r1 * X, (-1, 1)) + np.reshape(center, (-1, 1))
        K1 = cam_to_img[:2, :]
        assert R1.shape == (3, 2) and K1.shape == (2, 3)

        A = np.dot(K1, R1)
        C = np.dot(K1, b)
        assert A.shape == (2, 2) and C.shape == (2, 1)

        b3 = b[2]
        rT = R1[2, :]

        [y, x] = pix_pos2D
        is_corner = False
        for ind, corner in enumerate(self.corners2D):
            if x == corner[0] and y == corner[1] and ind in [0, 3, 4, 7]:
                # print("replacing {} and {} with".format(x, y))
                x = self.real_points2D[ind][0]
                y = self.real_points2D[ind][1]
                # print("{} and {}".format(x, y))
                is_corner = True
                corner_id = ind
                break

        left = np.reshape([x * b3, y * b3], (2, 1)) - C

        tmp = np.dot(np.reshape([x, y], (2, 1)), np.reshape(rT, (1, 2)))
        right = A - tmp

        ans = np.dot(np.linalg.inv(right), left)
        # print(ans)

        if is_corner:
            print("Corner {} loss: {}".format(
                corner_id, self.corners3D[corner_id] - [X, ans[0], ans[1]])
            )
            is_corner = False

        return ([X, ans[0], ans[1]])

    def __solve_right(self, pix_pos2D, R, dims, center, cam_to_img):

        # Prepare variables
        X = None
        Y = - dims[1] / 2  # -dy/2
        Z = None

        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]

        R1 = np.zeros(shape=(3, 2))
        R1[:, 0] = r1
        R1[:, 1] = r3

        b = np.reshape(r2 * Y, (-1, 1)) + np.reshape(center, (-1, 1))
        K1 = cam_to_img[:2, :]
        assert R1.shape == (3, 2) and K1.shape == (2, 3)

        A = np.dot(K1, R1)
        C = np.dot(K1, b)
        assert A.shape == (2, 2) and C.shape == (2, 1)

        b3 = b[2]
        rT = R1[2, :]

        [y, x] = pix_pos2D
        is_corner = False
        for ind, corner in enumerate(self.corners2D):
            if x == corner[0] and y == corner[1] and ind in [0, 1, 4, 5]:
                # print("replacing {} and {} with".format(x, y))
                x = self.real_points2D[ind][0]
                y = self.real_points2D[ind][1]
                # print("{} and {}".format(x, y))
                is_corner = True
                corner_id = ind
                break

        left = np.reshape([x * b3, y * b3], (2, 1)) - C

        tmp = np.dot(np.reshape([x, y], (2, 1)), np.reshape(rT, (1, 2)))
        right = A - tmp

        ans = np.dot(np.linalg.inv(right), left)
        # print(ans)

        if is_corner:
            print("Corner {} loss: {}".format(
                corner_id, self.corners3D[corner_id] - [ans[0], Y, ans[1]])
            )
            is_corner = False
        return ([ans[0], Y, ans[1]])

    def __solve_rear(self, pix_pos2D, R, dims, center, cam_to_img):

        # Prepare variables
        X = -dims[0] / 2  # -dx/2
        Y = None
        Z = None

        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]

        R1 = np.zeros(shape=(3, 2))
        R1[:, 0] = r2
        R1[:, 1] = r3

        b = np.reshape(r1 * X, (-1, 1)) + np.reshape(center, (-1, 1))
        K1 = cam_to_img[:2, :]
        assert R1.shape == (3, 2) and K1.shape == (2, 3)

        A = np.dot(K1, R1)
        C = np.dot(K1, b)
        assert A.shape == (2, 2) and C.shape == (2, 1)

        b3 = b[2]
        rT = R1[2, :]

        [y, x] = pix_pos2D
        is_corner = False
        for ind, corner in enumerate(self.corners2D):
            if x == corner[0] and y == corner[1] and ind in [1, 2, 5, 6]:
                # print("replacing {} and {} with".format(x, y))
                x = self.real_points2D[ind][0]
                y = self.real_points2D[ind][1]
                # print("{} and {}".format(x, y))
                is_corner = True
                corner_id = ind
                break

        left = np.reshape([x * b3, y * b3], (2, 1)) - C

        tmp = np.dot(np.reshape([x, y], (2, 1)), np.reshape(rT, (1, 2)))
        right = A - tmp

        ans = np.dot(np.linalg.inv(right), left)
        # print(ans)

        if is_corner:
            print("Corner {} loss: {}".format(
                corner_id, self.corners3D[corner_id] - [X, ans[0], ans[1]])
            )
            is_corner = False
        return [X, ans[0], ans[1]]

    def __solve_left(self, pix_pos2D, R, dims, center, cam_to_img):
        # Prepare variables
        X = None
        Y = dims[1] / 2  # dy/2
        Z = None

        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]

        R1 = np.zeros(shape=(3, 2))
        R1[:, 0] = r1
        R1[:, 1] = r3

        b = np.reshape(r2 * Y, (-1, 1)) + np.reshape(center, (-1, 1))
        K1 = cam_to_img[:2, :]
        assert R1.shape == (3, 2) and K1.shape == (2, 3)

        A = np.dot(K1, R1)
        C = np.dot(K1, b)
        assert A.shape == (2, 2) and C.shape == (2, 1)

        b3 = b[2]
        rT = R1[2, :]

        [y, x] = pix_pos2D
        is_corner = False
        for ind, corner in enumerate(self.corners2D):
            if x == corner[0] and y == corner[1] and ind in [2, 3, 6, 7]:
                # print("replacing {} and {} with".format(x, y))
                x = self.real_points2D[ind][0]
                y = self.real_points2D[ind][1]
                # print("{} and {}".format(x, y))
                is_corner = True
                corner_id = ind
                break

        left = np.reshape([x * b3, y * b3], (2, 1)) - C

        tmp = np.dot(np.reshape([x, y], (2, 1)), np.reshape(rT, (1, 2)))
        right = A - tmp

        ans = np.dot(np.linalg.inv(right), left)
        # print(ans)

        if is_corner:
            print("Corner {} loss: {}".format(
                corner_id, self.corners3D[corner_id] - [ans[0], Y, ans[1]])
            )
            is_corner = False
        return ([ans[0], Y, ans[1]])

    def __solve_roof(self, pix_pos2D, R, dims, center, cam_to_img):
        # Prepare variables
        X = None
        Y = None
        Z = dims[2] / 2  # dz/2

        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]

        R1 = np.zeros(shape=(3, 2))
        R1[:, 0] = r1
        R1[:, 1] = r2

        b = np.reshape(r3 * Z, (-1, 1)) + np.reshape(center, (-1, 1))
        K1 = cam_to_img[:2, :]
        assert R1.shape == (3, 2) and K1.shape == (2, 3)

        A = np.dot(K1, R1)
        C = np.dot(K1, b)
        assert A.shape == (2, 2) and C.shape == (2, 1)

        b3 = b[2]
        rT = R1[2, :]

        [y, x] = pix_pos2D
        is_corner = False
        for ind, corner in enumerate(self.corners2D):
            if x == corner[0] and y == corner[1] and ind in [4, 5, 6, 7]:
                # print("replacing {} and {} with".format(x, y))
                x = self.real_points2D[ind][0]
                y = self.real_points2D[ind][1]
                # print("{} and {}".format(x, y))
                is_corner = True
                corner_id = ind
                break

        left = np.reshape([x * b3, y * b3], (2, 1)) - C

        tmp = np.dot(np.reshape([x, y], (2, 1)), np.reshape(rT, (1, 2)))
        right = A - tmp

        ans = np.dot(np.linalg.inv(right), left)
        # print(ans)

        if is_corner:
            print("Corner {} loss: {}".format(
                corner_id, self.corners3D[corner_id] - [ans[0], ans[1], Z])
            )
            is_corner = False
        return (ans[0], ans[1], Z)

    def init_corners3D(self, dims):
        points3D = np.zeros((8, 3))
        # x y z
        x_len = dims[0] / 2
        y_len = dims[1] / 2
        z_len = dims[2] / 2

        x_s = [x_len, -x_len, -x_len, x_len, x_len, -x_len, -x_len, x_len]
        y_s = [-y_len, -y_len, y_len, y_len, -y_len, -y_len, y_len, y_len]
        z_s = [-z_len, -z_len, -z_len, -z_len, z_len, z_len, z_len, z_len]
        cnt = 0
        for cnt in range(8):
            points3D[cnt] = [x_s[cnt], y_s[cnt], z_s[cnt]]
            cnt += 1

        self.corners3D = points3D
        return points3D

    def gen_3D_box(self, R, dims, center, cam_to_img):

        corners3D = self.init_corners3D(dims)
        print("3D Corners: {}".format(corners3D))
        center = np.reshape(center, (-1, 1))
        corners2D = self.points3D_to_2D(corners3D, center, R, cam_to_img)
        corners2D = np.reshape(corners2D, (-1, 2))
        print("2D Corners: {}".format(corners2D))

        self.corners2D = corners2D
        return corners2D

    def points3D_to_2D(self, points3D, center, R, cam_to_img):
        points2D = []
        for point3D in points3D:
            point3D = point3D.reshape((-1, 1))
            point = center + np.dot(R, point3D)
            point = np.dot(cam_to_img, point)
            point2D = point[:2] / point[2]
            points2D.append(point2D)
        self.real_points2D = points2D
        points2D = np.asarray(points2D, int)
        return points2D

    def draw_3D_box(self, points2D, img):
        img = cv2.line(img, tuple(points2D[0]), tuple(points2D[1]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[1]), tuple(points2D[2]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[2]), tuple(points2D[3]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[3]), tuple(points2D[0]), (0, 0, 255), 2)

        # Vertical lines
        img = cv2.line(img, tuple(points2D[0]), tuple(points2D[4]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[1]), tuple(points2D[5]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[2]), tuple(points2D[6]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[3]), tuple(points2D[7]), (0, 0, 255), 2)

        img = cv2.line(img, tuple(points2D[4]), tuple(points2D[5]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[5]), tuple(points2D[6]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[6]), tuple(points2D[7]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(points2D[7]), tuple(points2D[4]), (0, 0, 255), 2)

        return img

    def find_visible_surfaces(self, yaw, dims, center, cam_to_img, img):

        x_len = dims[0] / 2
        y_len = dims[1] / 2
        z_len = dims[2] / 2

        # License, right door, Trunk, left door, roof
        x_normals = [x_len, 0, -x_len, 0, 0]
        y_normals = [0, -y_len, 0, y_len, 0]
        z_normals = [0, 0, 0, 0, z_len]

        normals = []
        cnt = 0
        for cnt in range(len(x_normals)):
            normals.append([x_normals[cnt], y_normals[cnt], z_normals[cnt]])
            cnt += 1

        visibilities = np.zeros(len(normals))
        for ind, norm in enumerate(normals):
            surf_normal = np.reshape(norm, (-1, 1))

            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            R = np.dot(self.cord_align, R)
            tmp = np.dot(R, surf_normal)

            # 3D Center of the surface
            surf3Dcenter = tmp + np.reshape(center, (-1, 1))

            # Visibility Vector = (t - (RxN + t) = -RxN)
            vis_vec = np.dot(-R, surf_normal)

            # 2D center of the surface (JUST For drawing)
            surf2D = np.dot(cam_to_img, surf3Dcenter)
            pixel = surf2D[:2] / surf2D[2]

            if np.inner(np.reshape(vis_vec, (1, 3)), np.reshape(surf3Dcenter, (1, 3))) > 0:
                visibilities[ind] = 1
                img = cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius=4, thickness=2, color=(0, 255, 0))
            else:
                visibilities[ind] = 0
                img = cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius=4, thickness=2, color=(0, 0, 255))

        return normals, visibilities, img

    def generate_2D_mask(self, img_shape, points2D, name='polygon'):
        mask = np.zeros(shape=img_shape, dtype=np.uint8)
        mask[points2D[:, 1], points2D[:, 0]] = 255

        polygon = convex_hull_image(mask).astype(int) * 255
        cv2.imwrite('{}.png'.format(name), polygon)

        return polygon

    def iterate_sides(self, visibilities, corners2D, yaw, dims, center, cam_to_img, img):
        for ind, visible in enumerate(visibilities):
            if not visible:
                continue
            else:
                if ind == 0:  # front
                    roi2D = [corners2D[0], corners2D[3], corners2D[4], corners2D[7]]
                    print('front visible!')
                elif ind == 1:  # right door
                    roi2D = [corners2D[0], corners2D[1], corners2D[4], corners2D[5]]
                    print('right door visible!')

                elif ind == 2:  # rear
                    roi2D = [corners2D[1], corners2D[2], corners2D[5], corners2D[6]]
                    print('rear visible!')

                elif ind == 3:  # left door
                    roi2D = [corners2D[2], corners2D[3], corners2D[6], corners2D[7]]
                    print('left door visible!')

                elif ind == 4:  # roof
                    roi2D = [corners2D[4], corners2D[5], corners2D[6], corners2D[7]]
                    print('roof visible!')

            roi2D = np.asarray(roi2D)
            mask = self.generate_2D_mask(img.shape, roi2D, 'side{}'.format(ind))

            img = self.handle_pixels(img, mask, yaw, dims, center, cam_to_img, ind)
            cv2.imwrite('sidecolored{}.png'.format(ind), img)

    def handle_pixels(self, img, mask, yaw, dims, center, cam_to_img, side):
        mask = np.asarray(mask)
        print('mask shape: ', mask.shape)
        mask = mask[:, :, 0]
        pois = np.where(mask == 255)

        pois = np.asarray(list(zip(pois[0], pois[1])))
        pois = np.reshape(pois, (-1, 2))
        color_step = 255 / len(pois)
        blue = 0
        r = self.rotate(yaw)
        for pix_pos2D in pois:
            img = cv2.circle(img, (int(pix_pos2D[1]), int(pix_pos2D[0])), radius=1,
                             thickness=1, color=(blue, 0, 255 - blue))
            pix_val = img[pix_pos2D[0], pix_pos2D[1]]
            pix_pos3D = self.solvers[side](pix_pos2D, r, dims, center, cam_to_img)
            blue += color_step
        return img

    def generate_side_mask(self, side, img_shape, corners2D):
        if side == 0:  # front
            roi2D = [corners2D[0], corners2D[3], corners2D[4], corners2D[7]]
        elif side == 1:  # right door
            roi2D = [corners2D[0], corners2D[1], corners2D[4], corners2D[5]]
        elif side == 2:  # rear
            roi2D = [corners2D[1], corners2D[2], corners2D[5], corners2D[6]]
        elif side == 3:  # left door
            roi2D = [corners2D[2], corners2D[3], corners2D[6], corners2D[7]]
        elif side == 4:  # roof
            roi2D = [corners2D[4], corners2D[5], corners2D[6], corners2D[7]]
        roi2D = np.asarray(roi2D)
        mask = self.generate_2D_mask(img_shape, roi2D, 'side{}'.format(side))
        return mask

    def check_in_mask(self, mask, point):
        if mask[point[0], point[1]] == 255:
            return True
        else:
            return False

    def find_key_points(self, img, mask):
        feature_params = dict(maxCorners=10000,
                              qualityLevel=0.2,
                              minDistance=7,
                              blockSize=7)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(img_gray, mask=None, **feature_params)

        results = []
        p0 = p0.reshape((-1, 2))
        image2 = img.copy()

        for ind, [x, y] in enumerate(p0):  # limiting keypoints' location to the polygon of the car
            if (mask[int(y), int(x), 0]) == 255:
                results.append(p0[ind])
                image2 = cv2.circle(image2, (int(x), int(y)), 4, (255, 255, 1), 1)

        return results

    def find_3Dlocation(self, kp, corners2D, visibilities, cam_to_img, yaw, dims, center):
        masks = []
        locations = []
        for ind, value in enumerate(visibilities):
            if value == 1:
                mask = self.generate_side_mask(ind, img.shape, np.array(corners2D))
                masks.append((ind, mask[:, :, 0]))
        r = self.rotate(yaw)
        for keypoint in kp:
            [x, y] = keypoint
            for side, mask in masks:
                if self.check_in_mask(mask, (int(y), int(x))):
                    location = utils.solvers[side]([int(y), int(x)], r, dims, center, cam_to_img)
                    locations.append([float(location[0]), float(location[1]), float(location[2])])
                    break
        return locations

    def track_points(self, old_frame, p0, new_frame, mask, keypoints_3Dlocation):
        lk_params = dict(winSize=(25, 25),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # print(st.shape)
        # # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        print(good_new)

        for i, (new, old) in enumerate(zip(p0, p1)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            new_frame = cv2.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(new_frame, mask)
        cv2.imshow('frame', img)
        cv2.waitKey()

        return p1, mask

    def drawit(self, img, points2D):

        img = cv2.line(img, (points2D[0][0], points2D[0][1]), (points2D[1][0], points2D[1][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[1][0], points2D[1][1]), (points2D[2][0], points2D[2][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[2][0], points2D[2][1]), (points2D[3][0], points2D[3][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[3][0], points2D[3][1]), (points2D[0][0], points2D[0][1]), (0, 0, 255), 2)
        # Vertic,
        img = cv2.line(img, (points2D[0][0], points2D[0][1]), (points2D[4][0], points2D[4][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[1][0], points2D[1][1]), (points2D[5][0], points2D[5][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[2][0], points2D[2][1]), (points2D[6][0], points2D[6][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[3][0], points2D[3][1]), (points2D[7][0], points2D[7][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[4][0], points2D[4][1]), (points2D[5][0], points2D[5][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[5][0], points2D[5][1]), (points2D[6][0], points2D[6][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[6][0], points2D[6][1]), (points2D[7][0], points2D[7][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points2D[7][0], points2D[7][1]), (points2D[4][0], points2D[4][1]), (0, 0, 255), 2)

        return img

    def generate_final_R(self, yaw):
        rot_M = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.finalR = np.dot(self.cord_align, rot_M)
        return self.finalR

    def update_final_R(self, R):
        self.finalR = np.dot(R, self.finalR)
        return self.finalR

    def points3D_to_2D2(self, points3D, center, R, cam_to_img):
        print(points3D.shape)
        point3D = points3D.reshape((3, -1))
        point = center + np.dot(R, point3D)
        point = np.dot(cam_to_img, point)
        point2D = point[:2] / point[2]
        self.real_points2D = point2D
        points2D = np.asarray(point2D, int)

        return points2D

    def draw_point_corespondence(self, old_frame, p0, new_frame, p1):
        old_frame = old_frame.copy()
        new_frame = new_frame.copy()
        for i in range (p0.shape[0]):
            x1,y1 = p0[i]
            x2,y2 = p1[i]
            color = tuple(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))
            old_frame = cv2.circle(old_frame, (int(x1),int(y1)), 2, color,2)
            new_frame = cv2.circle(new_frame, (int(x2),int(y2)), 2,color,2)
        cv2.imshow("old",old_frame)
        cv2.imshow("new",new_frame)





if __name__ == "__main__":
    cap = cv2.VideoCapture('drive 3.avi')

    utils = Utils()

    # X Y Z in cam coordinates
    # X goes right, Y goes down, Z gets far
    # center = [-2, 0, 10]
    center = [3.20864, 1.51651, 8.07392]
    # X Y Z in obj coordinates
    # X goes through head lights, Y goes through driver's door, Z goes through roof.
    # dims = [4, 4, 2]
    # dim length, dim width, dim height
    dims = [4.1015, 1.51056, 1.38166]
    # center[1] = center[1] - dims[1]/2
    # Rotation Angle(Rad) along Object's Z (from center to roof)
    # yaw = np.pi/2
    center[1] = center[1] - dims[2] / 2
    yaw = -1.55983 + np.pi / 2
    # The 3x3 K matrix
    calib_mat = np.array([
        [7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
        [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
        [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
    ])

    # calib_mat = np.array([[718, 0., 607],
    #                     [0., 718, 185],
    #                     [0., 0., 1.]]) 

    ret, img = cap.read()
    # ret, img = cap.read()
    image = np.copy(img)
    R = utils.rotate(yaw)
    corners2D = utils.gen_3D_box(R, dims, center, calib_mat)

    mask = utils.generate_2D_mask(img.shape, corners2D)

    img = utils.draw_3D_box(corners2D, img)

    normals, visibilities, img = utils.find_visible_surfaces(yaw, dims, center, calib_mat, img)

    utils.iterate_sides(visibilities, corners2D, yaw, dims, center, calib_mat, img.copy())
    # Write the output image
    cv2.imwrite('test.png', image)

    keypoints = utils.find_key_points(image, mask)
    for [x, y] in keypoints:
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), 3)

    keypoints_3Dlocation = utils.find_3Dlocation(keypoints, corners2D, visibilities, calib_mat, yaw, dims, center)

    cv2.imshow("f", img)
    cv2.waitKey()
    ret, new_frame = cap.read()

    old_frame = np.copy(image)
    draw_mask = np.zeros_like(old_frame)

    p0 = np.array(keypoints)
    p0 = p0.reshape(-1, 1, 2)
    keypoints_3Dlocation = np.array(keypoints_3Dlocation)
    dist_coeffs = np.zeros((4, 1))
    finalR = utils.generate_final_R
    while 1:
        ret, new_frame = cap.read()
        p1, draw_mask = utils.track_points(old_frame, p0, new_frame, draw_mask, keypoints_3Dlocation)
        utils.draw_point_corespondence(old_frame, p0.reshape(-1,2), new_frame, p1.reshape(-1,2))
        p0 = p1
        old_frame = new_frame

        success, rvecs, tvecs, inliners = cv2.solvePnPRansac(keypoints_3Dlocation, p1.reshape(-1, 2), calib_mat,
                                                             dist_coeffs)
        rot, _ = cv2.Rodrigues(rvecs)
        points_2D = utils.points3D_to_2D2(utils.corners3D.T, tvecs, rot, calib_mat)
        points_2D = points_2D.astype(int)
        image = utils.drawit(new_frame, points_2D.T)
        cv2.imshow("image", new_frame)
