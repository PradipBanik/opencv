import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_color = cv2.imread('./kitten.jpg', 1)
# kernel = np.ones((11,11), np.float32)*(1/5**2)
#
# # blurr = cv2.filter2D(img_color, -1, kernel)
# # blurr = cv2.blur(img_color, (5,5))
# # blurr = cv2.GaussianBlur(img_color,(11,11),0, 0, 0)
# # blurr_sigma5 = cv2.GaussianBlur(img_color,(11,11),5)
#
# #sharpening
# kernel_sarp = np.array(
#                         [[0,-100,0],
#                          [-100,401,-100],
#                          [0,-100,0]])
#
# sharpen = cv2.filter2D(img_color, -1, kernel_sarp)
#
# cv2.imshow('original', sharpen)
# cv2.imshow('frame', img_color)
# # print(blurr)
# cv2.waitKey(0)

# im_image = cv2.imread('./checkerboard_color.png', 1)
# im_image_grey = cv2.cvtColor(im_image, cv2.COLOR_BGR2GRAY)
# kernel = np.array([[-1,0,-1],
#                    [-2,0,2],
#                    [-1,0,1]])
# image = np.ones((8,8), np.uint8) *90
# image[1:7,1:4] = 20
# image[1:7,4:7] = 150
#
# # conv = cv2.filter2D(im_image_grey, cv2.CV_64F, kernel)#, borderType=cv2.BORDER_REPLICATE)
# # cov = conv.astype(np.uint8)
# # print(image)
# convx = cv2.Sobel(im_image_grey, cv2.CV_64F, 1, 0, ksize=5)
# convy = cv2.Sobel(im_image_grey, cv2.CV_64F, 0, 1, ksize=3)
#
# # print(conv)
# plt.plot(1,1,1)
# plt.imshow(im_image_grey, cmap='gray')
# plt.show()
# plt.imshow(convy, cmap='gray')
# plt.show()

# im_color =cv2.imread('./butterfly.jpg', 1)
# im_gray = cv2.cvtColor(im_color,cv2.COLOR_BGR2GRAY)
# im_blur = cv2.GaussianBlur(im_gray,(5,5), 1.4)
# canny = cv2.Canny(im_blur, 50, 200, apertureSize=3)
# plt.imshow(canny, cmap='gray')
# plt.show()
frame_gray = 0
button = 0
keepc = 0
keepg = 0
keeps = 0
keepf = 0
keepd = 0
keepb = 0
keepdd = 0
keeppn = 0
keepsy = 0
def units(keec, keeg, kees, keef, keed, keeb, keedd, keepn, keesy):
    # global button
    global keepc
    global keepg
    global keeps
    global keepf
    global keepd
    global keepb
    global keepdd
    global keeppn
    global keepsy
    # button = butto
    keepc = keec
    keepg = keeg
    keeps= kees
    keepf = keef
    keepd = keed
    keepb = keeb
    keepdd = keedd
    keeppn = keepn
    keepsy = keesy

def buttons(button, frame):
    button = ord(button)
    frame = cv2.resize(frame, (1000,900))
    # global frame_gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame_gray, (5, 5), 1.4)
    dis = frame.copy()
    if button ==ord('r'):
        units(0,0,0,0,0,0, 0, 0, 0)
        # keepc = 0
        # keepg = 0
        # keeps = 0
        # keepf = 0
        # keepd = 0
        # keepb = 0

    if button==ord('c') or keepc ==1:

        canny_frame = cv2.Canny(blurred_frame, 50, 60)
        dis = canny_frame.copy()
        dis = cv2.cvtColor(dis, cv2.COLOR_GRAY2BGR)
        units(1,0,0,0,0,0, 0, 0, 0)

        # keepc = 1
        # keepg = 0
        # keeps = 0
        # keepf = 0
        # keepd = 0
        # keepb = 0

    if button ==ord('g') or keepg ==1:

        dis = frame_gray.copy()
        dis = cv2.cvtColor(dis, cv2.COLOR_GRAY2BGR)
        units(0,1,0,0,0,0, 0, 0, 0)

        # keepc = 0
        # keepg = 1
        # keeps = 0
        # keepf = 0
        # keepd = 0
        # keepb = 0

    if button ==ord('s') or keeps ==1:
        frame_sepia = frame.copy()
        frame_sepia = cv2.resize(frame_sepia, (frame.shape[1], frame.shape[0]))
        frame_seapia_rgb = cv2.cvtColor(frame_sepia, cv2.COLOR_BGR2RGB)
        frame_seapia_rgb_float = np.array(frame_seapia_rgb, np.float64)
        frame_seapia_rgb_float_tf = cv2.transform(frame_seapia_rgb_float,
                                                  np.matrix([[0.390, 0.760, 0.189],
                                                       [0.350, 0.680, 0.160],
                                                        [0.270, 0.530, 0.130]]))
        frame_seapia_rgb_float_tf = np.clip(frame_seapia_rgb_float_tf,0,255)
        frame_seapia_rgb_float_tf = np.array(frame_seapia_rgb_float_tf, np.uint8)
        frame_seapia_rgb_float_tf_bgr = cv2.cvtColor(
            frame_seapia_rgb_float_tf, cv2.COLOR_RGB2BGR)

        dis = frame_seapia_rgb_float_tf_bgr.copy()
        units(0,0,1,0,0,0, 0, 0, 0)

        # keepc = 0
        # keepg = 0
        # keeps=1
        # keepf = 0
        # keepd = 0
        # keepb = 0

        # keepd = 1

    if button ==ord('f') or keepf ==1:
        frame_gauss = frame.copy()
        gausian_kernelx = cv2.getGaussianKernel(frame.shape[1], frame.shape[1]/2)
        gausian_kernely = cv2.getGaussianKernel(frame.shape[0], frame.shape[0]/2)

        kernel = gausian_kernelx.transpose()*gausian_kernely
        mask = kernel/kernel.max()

        for i in range(0,3):
            frame_gauss[:,:,i] = frame_gauss[:,:,i]*mask

        dis = frame_gauss.copy()
        # print(gausian_kernelx)

        units(0,0,0,1,0,0, 0, 0, 0)

        # keepc = 0
        # keepg = 0
        # keeps = 0
        # keepf = 1
        # keepd = 0
        # keepb = 0

    if button == ord('d') or keepd == 1:
        kernel = np.array(
            [[0,-3,-3],
             [3,0,-3],
             [3,3,0]]
        )
        d_effect = cv2.filter2D(frame,-1,kernel)
        dis = d_effect.copy()

        units(0,0,0,0,1,0, 0, 0, 0)

        # keepc = 0
        # keepg = 0
        # keeps = 0
        # keepf = 0
        # keepd = 1
        # keepb = 0

    if button == ord('b') or keepb == 1:
        brigten = frame.copy()
        brigten = cv2.convertScaleAbs(brigten, alpha = 2, beta=25)
        dis = brigten.copy()
        units(0,0,0,0,0,1, 0, 0, 0)

        # keepc = 0
        # keepg = 0
        # keeps = 0
        # keepf = 0
        # keepd = 0
        # keepb = 1

    if button == ord('l') or keepdd == 1:
        k=9
        border = frame.copy()
        kernel = np.array(
            [[-1,-1,-1],
             [-1,k,-1],
             [-1,-1,-1]]
        )
        border_d =cv2.filter2D(border, -1, kernel)
        dis = border_d.copy()
        units(0, 0, 0, 0, 0, 0,1, 0, 0)

    if button == ord('n') or keeppn == 1:
        pencil_sk = frame.copy()
        pencil_sk_blur =cv2.GaussianBlur(pencil_sk, (5,5), 0)
        # pencil_sk = pencil_sk.astype(np.uint8)
        # print(pencil_sk.shape)
        pencil_sk_op, pencil_sk_2 = cv2.pencilSketch(pencil_sk_blur)
        dis = pencil_sk_2.copy()
        units(0, 0, 0, 0, 0, 0,0, 1, 0)

    if button == ord('t') or keepsy == 1:
        style = frame.copy()
        style_blur =cv2.GaussianBlur(style, (5,5), 0)
        style_final = cv2.stylization(style_blur, sigma_s=30, sigma_r=1)
        dis = style_final.copy()
        units(0, 0, 0, 0, 0, 0,0, 0, 1)

    return dis

def video_loop():
    flower = cv2.imread('./Applications/Flowers.jpg')
    house = cv2.imread('./Applications/House.jpg')
    city = cv2.imread('./Applications/New_York.jpg')
    monument = cv2.imread('./Applications/Monument.jpg')
    preview = 0
    canny = 1

    global button
    global keepc
    global keepg
    global keeps
    global keepf
    global keepd
    global keepb
    global keepd
    global keeppn
    global keepsy
    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        frame_exist, frame = video_capture.read()

        if not frame_exist:
            break
        frame = cv2.flip(frame, 1)


        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(frame_gray, (5, 5), 1.4)

        # thresh = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 3,9)


        # # 4. Canny with auto-thresholds
        # median = np.median(blurred_frame)
        # sigma = 0.33  # Standard deviation multiplier
        # low = int(max(0, (1.0 - sigma) * median))
        # high = int(min(255, (1.0 + sigma) * median))

        # canny_frame = cv2.Canny(blurred_frame, 50, 60)
        # canny_frame = cv2.Canny(frame_gray, 145, 150)

        # print(canny_frame)

        # 5. Optional: Morphological cleanup
        # kernel = np.ones((2, 2), np.uint8)
        # edges = cv2.morphologyEx(canny_frame, cv2.MORPH_CLOSE, kernel)
        print(button)
        dis = frame.copy()
        if button ==ord('r'):
            units(0,0,0,0,0,0, 0, 0, 0)
            # keepc = 0
            # keepg = 0
            # keeps = 0
            # keepf = 0
            # keepd = 0
            # keepb = 0

        if button==ord('c') or keepc ==1:
            canny_frame = cv2.Canny(blurred_frame, 50, 60)
            dis = canny_frame.copy()
            units(1,0,0,0,0,0, 0, 0, 0)

            # keepc = 1
            # keepg = 0
            # keeps = 0
            # keepf = 0
            # keepd = 0
            # keepb = 0

        if button ==ord('g') or keepg ==1:
            dis = frame_gray.copy()
            units(0,1,0,0,0,0, 0, 0, 0)

            # keepc = 0
            # keepg = 1
            # keeps = 0
            # keepf = 0
            # keepd = 0
            # keepb = 0


        if button ==ord('s') or keeps ==1:
            frame_sepia = frame.copy()
            frame_sepia = cv2.resize(frame_sepia, (frame.shape[1], frame.shape[0]))
            frame_seapia_rgb = cv2.cvtColor(frame_sepia, cv2.COLOR_BGR2RGB)
            frame_seapia_rgb_float = np.array(frame_seapia_rgb, np.float64)
            frame_seapia_rgb_float_tf = cv2.transform(frame_seapia_rgb_float,
                                                      np.matrix([[0.390, 0.760, 0.189],
                                                           [0.350, 0.680, 0.160],
                                                            [0.270, 0.530, 0.130]]))
            frame_seapia_rgb_float_tf = np.clip(frame_seapia_rgb_float_tf,0,255)
            frame_seapia_rgb_float_tf = np.array(frame_seapia_rgb_float_tf, np.uint8)
            frame_seapia_rgb_float_tf_bgr = cv2.cvtColor(
                frame_seapia_rgb_float_tf, cv2.COLOR_RGB2BGR)

            dis = frame_seapia_rgb_float_tf_bgr.copy()
            units(0,0,1,0,0,0, 0, 0, 0)

            # keepc = 0
            # keepg = 0
            # keeps=1
            # keepf = 0
            # keepd = 0
            # keepb = 0

            # keepd = 1

        if button ==ord('f') or keepf ==1:
            frame_gauss = frame.copy()
            gausian_kernelx = cv2.getGaussianKernel(frame.shape[1], frame.shape[1]/2)
            gausian_kernely = cv2.getGaussianKernel(frame.shape[0], frame.shape[0]/2)

            kernel = gausian_kernelx.transpose()*gausian_kernely
            mask = kernel/kernel.max()

            for i in range(0,3):
                frame_gauss[:,:,i] = frame_gauss[:,:,i]*mask

            dis = frame_gauss.copy()
            # print(gausian_kernelx)

            units(0,0,0,1,0,0, 0, 0, 0)

            # keepc = 0
            # keepg = 0
            # keeps = 0
            # keepf = 1
            # keepd = 0
            # keepb = 0

        if button == ord('d') or keepd == 1:
            kernel = np.array(
                [[0,-3,-3],
                 [3,0,-3],
                 [3,3,0]]
            )
            d_effect = cv2.filter2D(frame,-1,kernel)
            dis = d_effect.copy()

            units(0,0,0,0,1,0, 0, 0, 0)

            # keepc = 0
            # keepg = 0
            # keeps = 0
            # keepf = 0
            # keepd = 1
            # keepb = 0

        if button == ord('b') or keepb == 1:
            brigten = frame.copy()
            brigten = cv2.convertScaleAbs(brigten, alpha = 2, beta=25)
            dis = brigten.copy()
            units(0,0,0,0,0,1, 0, 0, 0)

            # keepc = 0
            # keepg = 0
            # keeps = 0
            # keepf = 0
            # keepd = 0
            # keepb = 1

        if button == ord('l') or keepdd == 1:
            k=9
            border = frame.copy()
            kernel = np.array(
                [[-1,-1,-1],
                 [-1,k,-1],
                 [-1,-1,-1]]
            )
            border_d =cv2.filter2D(border, -1, kernel)
            dis = border_d.copy()
            units(0, 0, 0, 0, 0, 0,1, 0, 0)

        if button == ord('n') or keeppn == 1:
            pencil_sk = frame.copy()
            pencil_sk_blur =cv2.GaussianBlur(pencil_sk, (5,5), 0)
            # pencil_sk = pencil_sk.astype(np.uint8)
            # print(pencil_sk.shape)
            pencil_sk_op, pencil_sk_2 = cv2.pencilSketch(pencil_sk_blur)
            dis = pencil_sk_op.copy()
            units(0, 0, 0, 0, 0, 0,0, 1, 0)

        if button == ord('t') or keepsy == 1:
            style = frame.copy()
            style_blur =cv2.GaussianBlur(style, (5,5), 0)
            style_final = cv2.stylization(style_blur, sigma_s=30, sigma_r=1)
            dis = style_final.copy()
            units(0, 0, 0, 0, 0, 0,0, 0, 1)






        frame_name = 'frame'
        cv2.namedWindow(frame_name)

        cv2.imshow(frame_name, dis)
        button = cv2.waitKey(1)
        if button==ord('q'):
            break

if __name__ == "__main__":
    # if __name__ == "__main__":
    print(f"i am in main block")
    video_loop()
    cv2.destroyAllWindows()
