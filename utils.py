import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def data_maker():
    colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pandas.read_csv('driving_log.csv', skiprows=[0], names=colnames)
    center = data.center.tolist()
    center_recover = data.center.tolist() 
    left = data.left.tolist()
    right = data.right.tolist()
    steering = data.steering.tolist()
    steering_recover = data.steering.tolist()

    ## SPLIT TRAIN AND VALID ##flag
    #  Shuffle center and steering. Use 10% of central images and steering angles for validation.
    center, steering = shuffle(center, steering)
    center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 

    ## FILTER STRAIGHT, LEFT, RIGHT TURNS ## 
    #  (d_### is list of images name, a_### is list of angles going with list)
    d_straight, d_left, d_right = [], [], []
    a_straight, a_left, a_right = [], [], []
    for i in steering:
      #Positive angle is turning from Left -> Right. Negative is turning from Right -> Left#
      index = steering.index(i)
      if i > 0.15:
        d_right.append(center[index])
        a_right.append(i)
      if i < -0.15:
        d_left.append(center[index])
        a_left.append(i)
      else:
        d_straight.append(center[index])
        a_straight.append(i)

    ## ADD RECOVERY ##
    #  Find the amount of sample differences between driving straight & driving left, driving straight & driving right #
    ds_size, dl_size, dr_size = len(d_straight), len(d_left), len(d_right)
    main_size = math.ceil(len(center_recover))
    l_xtra = ds_size - dl_size
    r_xtra = ds_size - dr_size
    # Generate random list of indices for left and right recovery images
    indice_L = random.sample(range(main_size), l_xtra)
    indice_R = random.sample(range(main_size), r_xtra)

    # Filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
    for i in indice_L:
      if steering_recover[i] < -0.15:
        d_left.append(right[i])
        a_left.append(steering_recover[i] - FLAGS.steering_adjustment)

    # Filter angle more than 0.15 and add left camera images into driving right list, add an adjustment angle #  
    for i in indice_R:
      if steering_recover[i] > 0.15:
        d_right.append(left[i])
        a_right.append(steering_recover[i] + FLAGS.steering_adjustment)

    ## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##  
    X_train = d_straight + d_left + d_right
    y_train = np.float32(a_straight + a_left + a_right)
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    # print(data_dir)
    # print(image_file.strip())
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    # print(data_dir)
    choice = np.random.choice(3)
    # print("choice")
    if choice == 0:
        return load_image(data_dir, left), float(steering_angle) + 0.2
    elif choice == 1:
        return load_image(data_dir, right), float(steering_angle) - 0.2
    return load_image(data_dir, center), float(steering_angle)


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    choice = np.random.choice(4)
    # print("choice",choice)
    if choice == 0 :
        image= load_image(data_dir, left)
        steering_angle = steering_angle + 0.25
    elif choice == 1:
        image= load_image(data_dir, right)
        steering_angle = steering_angle - 0.25
    else:
        image= load_image(data_dir, center)
        steering_angle = steering_angle 
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # print("type",type(steering_angle))
            steering_angle=float(steering_angle)
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

