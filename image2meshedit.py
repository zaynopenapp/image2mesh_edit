import cv2
import numpy as np

# load image
img = cv2.imread("face.png")
if img is None:
    print("image not found")
    exit()

img = img.astype(np.float32)

h, w = img.shape[:2]

# ukuran grid mesh
grid_x = 8
grid_y = 8

points = []
orig_points = []

# membuat grid vertex
for y in range(grid_y + 1):
    for x in range(grid_x + 1):

        px = x * w / grid_x
        py = y * h / grid_y

        points.append([px, py])
        orig_points.append([px, py])

points = np.float32(points)
orig_points = np.float32(orig_points)

# triangulasi mesh
triangles = []

for y in range(grid_y):
    for x in range(grid_x):

        i = y * (grid_x + 1) + x

        p0 = i
        p1 = i + 1
        p2 = i + (grid_x + 1)
        p3 = i + (grid_x + 1) + 1

        triangles.append([p0, p1, p3])
        triangles.append([p0, p3, p2])


selected = -1


def warp_triangle(src_tri, dst_tri, img, out):

    r1 = cv2.boundingRect(np.float32(src_tri))
    r2 = cv2.boundingRect(np.float32(dst_tri))

    src_rect = []
    dst_rect = []

    for i in range(3):

        src_rect.append([
            src_tri[i][0] - r1[0],
            src_tri[i][1] - r1[1]
        ])

        dst_rect.append([
            dst_tri[i][0] - r2[0],
            dst_tri[i][1] - r2[1]
        ])

    src_rect = np.float32(src_rect)
    dst_rect = np.float32(dst_rect)

    img_crop = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    M = cv2.getAffineTransform(src_rect, dst_rect)

    warped = cv2.warpAffine(
        img_crop,
        M,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_rect), 1.0)

    mask = cv2.merge([mask, mask, mask])

    out_slice = out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    # samakan ukuran jika ada selisih pixel
    h1 = min(out_slice.shape[0], warped.shape[0])
    w1 = min(out_slice.shape[1], warped.shape[1])

    out_part = out_slice[:h1, :w1]
    warped = warped[:h1, :w1]
    mask = mask[:h1, :w1]

    out[r2[1]:r2[1]+h1, r2[0]:r2[0]+w1] = out_part*(1-mask) + warped*mask


def warp():

    out = np.zeros_like(img)

    for tri in triangles:

        src_tri = np.float32([orig_points[i] for i in tri])
        dst_tri = np.float32([points[i] for i in tri])

        warp_triangle(src_tri, dst_tri, img, out)

    return np.uint8(out)


def mouse(event, x, y, flags, param):

    global selected

    if event == cv2.EVENT_LBUTTONDOWN:

        for i, p in enumerate(points):

            if np.linalg.norm(p - [x, y]) < 10:
                selected = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:

        if selected != -1:
            points[selected] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:

        selected = -1


cv2.namedWindow("mesh")
cv2.setMouseCallback("mesh", mouse)


while True:

    frame = warp()

    # gambar mesh
    for tri in triangles:

        p1 = tuple(points[tri[0]].astype(int))
        p2 = tuple(points[tri[1]].astype(int))
        p3 = tuple(points[tri[2]].astype(int))

        cv2.line(frame, p1, p2, (0,255,0), 1)
        cv2.line(frame, p2, p3, (0,255,0), 1)
        cv2.line(frame, p3, p1, (0,255,0), 1)

    # gambar vertex
    for p in points:
        cv2.circle(frame, tuple(p.astype(int)), 3, (0,255,0), -1)

    cv2.imshow("mesh", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()