import carla
import time #pt delay
import cv2
import numpy as np
import math
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from numpy import ndarray
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import  subprocess

sys.path.append(r'C:\\Users\\mldma\\Desktop\\CARLA_0.9.14\\WindowsNoEditor\\PythonAPI\\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    #img = cv2.resize(img, (imageDimesions[0], imageDimesions[1]))
    img = img/255
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'

def process_image(image):
   # image.convert(carla.ColorConverter.Raw)
    #img = np.array(image.raw_data).reshape((image.height, image.width, 4))
    img = np.array(image, dtype=np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]  # Eliminăm canalul al patrulea (alpha)

    if len(img.shape) == 2:
    # Imagine grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convertim imaginea grayscale în RGB
    return img

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplică preprocesarea și predicția
        processed_frame = preprocessing(frame)
        processed_frame = cv2.resize(processed_frame, (32, 32))
        processed_frame = np.expand_dims(processed_frame, axis=0)
        prediction = model.predict(processed_frame)
        class_index = np.argmax(prediction)

        # Afișează rezultatele pe cadru
        label = getClassName(class_index)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video with Traffic Sign Recognition', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
 #   cv2.destroyAllWindows()

def maintain_speed(s):
    if s>= PREFERED_SPEED:
        return 0
    elif s<PREFERED_SPEED - SPEED_THRESHOLD:
        return 0.8 #procent
    else:
        return 0.4

def angle_between(v1,v2):
    return math.degrees(np.arctan2(v1[1],v1[0]) - np.arctan2(v2[1],v2[0]))

def get_angle(car, wp):
    # direction to selected waypoint
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y

    # vector to wp
    x = (wp_x - car_x) / ((wp_y - car_y) ** 2 + (wp_x - car_x) ** 2) ** 0.5
    y = (wp_y - car_y) / ((wp_y - car_y) ** 2 + (wp_x - car_x) ** 2) ** 0.5

    # car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = angle_between((x, y), (car_vector.x, car_vector.y))

    return degrees

def camera_callback(image,data_dict):
    global model, video_out
    data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
    image_np = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    image_np = image_np[:, :, :3]  # Luăm doar canalele RGB

    image_np = np.copy(image_np)
    # Preprocesarea imaginii pentru model
    processed_img = preprocessing(image_np)
    processed_img = cv2.resize(processed_img, (32, 32))  # Redimensionează la dimensiunea așteptată de model
    processed_img = np.expand_dims(processed_img, axis=0)  # Adaugă o dimensiune batch

    # Aplică modelul de recunoaștere
    predictions = model.predict(processed_img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    # Afisare rezultate pe imagine
    if probabilityValue > threshold:
        label = getClassName(classIndex)
        cv2.putText(image_np, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    video_out.write(image_np)
    # Afișează imaginea
    #cv2.imshow("Camera Feed", image_np)
    #cv2.waitKey(1)

def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def convert_video(input_file, output_file):
    input_file = "C:\\Users\\mldma\\Desktop\\CARLA_0.9.14\\WindowsNoEditor\\myData\\output_video.avi"
    output_file = "C:\\Users\\mldma\\Desktop\\CARLA_0.9.14\\WindowsNoEditor\\myData\\converted_video.mp4"
    command = f'ffmpeg -i "{input_file}" -c:v libx264 "{output_file}"'
    try:
        process =  subprocess.run(command, check = True, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        print(f"Video converted successfully")
        print(process.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("failed to convert")
        print(e.stderr.decode())

if __name__ == "__main__":
    print('Test')

    ffmpeg_path = "C:\\ffmpeg\\ffmpeg\\bin"
    if ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_path

    # Verifică dacă calea a fost setată
    #print(os.environ["PATH"])

    # Încearcă să rulezi o comandă ffmpeg pentru a vedea dacă este recunoscută
  #  os.system("ffmpeg -version")

    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    client.load_world('Town05')
    video_path = 'C:\\Users\\mldma\\Desktop\\CARLA_0.9.14\\WindowsNoEditor\\myData'
    video_filename = 'output_video.avi'
    #video_filename = 'nou.mp4'
    full_video_path = os.path.join(video_path, video_filename)

    # Asigură-te că directorul există
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    # Setează parametrii pentru înregistrarea video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(full_video_path, fourcc, 20.0, (640, 360))

    os.environ['KERAS_BACKEND'] = 'tensorflow'
    threshold = 0.75
    # Încărcare model

    model = load_model("..\\model_trained.h5")
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = world.get_blueprint_library().filter('*mini*')
    process_video(full_video_path,model)

    start_point = spawn_points[0]
    vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)

    for sp in spawn_points:
        if vehicle is not None:
            break
        else:
            vehicle = world.try_spawn_actor(vehicle_bp[0],sp)
    if vehicle is None:
        print("Vehicle could not be spawned")
        sys.exit(1)


    CAMERA_POS_Z = 3
    CAMERA_POS_X = -5

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '360')
    camera_bp.set_attribute('fov', '110')

    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    image_w = camera_bp.get_attribute('image_size_x').as_int()
    image_h = camera_bp.get_attribute('image_size_y').as_int()

    camera_data = {'image': np.zeros((image_h, image_w, 4))}
    camera.listen(lambda image: camera_callback(image, camera_data))
    # camera.listen(lambda image: camera_callback(image))

    PREFERED_SPEED = 30 #bun pt cand o sa fie semnele de circulatie
    SPEED_THRESHOLD = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30,30) #current speed cout
    org2 = (30,50) #steering angle
    org3 = (30,70)
    org4 = (30,90)
    org5 = (30,110)

    fontScale = 0.5

    color = (255,255,255) #alb
    thickness = 1

    #traseu colorat
    point_a = start_point.location
    sampling_resolution = 1
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)

    distance = 0
    route=[]
    for loc in spawn_points:

        cur_route = grp.trace_route(point_a, loc.location)
        if len(cur_route) > distance:
            distance = len(cur_route)
            route = cur_route

    for waypoint in route:
        world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=600.0,
                                persistent_lines=True)

    if not route:
        print("no route found")
        sys.exit(1)

    #afisare1
    cv2.namedWindow('rgb camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('rgb camera', camera_data['image'])
    vehicle.set_autopilot(True)
    quit = False
    curr_wp = 5
    predicted_angle = 0

    while curr_wp < len(route) - 1:
        world.tick()
        if cv2.waitKey(1) == ord('q'):
            quit = True
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

            break
        image = camera_data['image']

        if curr_wp >= len(route):
            print(f"Index out of range: curr_wp = {curr_wp}, route length = {len(route)}")
            break
        while curr_wp < len(route) and vehicle.get_transform().location.distance(
                route[curr_wp][0].transform.location) < 5:
            curr_wp += 1  # move to next wp

        predicted_angle = get_angle(vehicle, route[curr_wp][0])
        while curr_wp < len(route) and vehicle.get_transform().location.distance(
                route[curr_wp][0].transform.location) < 5:
            curr_wp += 1

        if curr_wp >= len(route):
            print(f"Index out of range after increment: curr_wp = {curr_wp}, route length = {len(route)}")
            break
        image = cv2.putText(image, 'steering angle: ' + str(round(predicted_angle, 3)), org, font, fontScale, color,
                            thickness, cv2.LINE_AA)
        v = vehicle.get_velocity()
        speed = round(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), 0)
        #   image = cv2.putText(image,'speed: '+str(int(speed))+'km/h',org2,font,fontScale,color,thickness,cv2.LINE_AA)

        fwd_vector = vehicle.get_transform().get_forward_vector()
        image = cv2.putText(image, 'speed: ' + str(int(speed)), org2, font, fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, 'next wp: ' + str(curr_wp), org3, font, fontScale, color, thickness, cv2.LINE_AA)
        estimated_throttle = maintain_speed(speed)
        if predicted_angle < -300:
            predicted_angle = predicted_angle + 360
        elif predicted_angle > 300:
            predicted_angle = predicted_angle - 360
        steer_input = predicted_angle

        if predicted_angle < -40:
            steer_input = -40
        elif predicted_angle > 40:
            steer_input = 40

        steer_input = steer_input / 75
        vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))

        cv2.imshow('rgb camera', image)
    cv2.destroyAllWindows()
    camera.stop()

   # for sensor in world.get_actors().filter('*sensor*'):
    #    sensor.destroy()
    #vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

    #convert_video(full_video_path,full_video_path.replace('.avi','.mp4'))

    os.environ['KERAS_BACKEND'] = 'tensorflow'

    frameWidth = 640
    frameHeight = 480
    brightness = 180
    threshold = 0.90
    font = cv2.FONT_HERSHEY_SIMPLEX

    sys.path.append('C:\\Users\\mldma\\Desktop\\CARLA_0.9.14\\WindowsNoEditor\\myData')

    path = "myData"
    #path = os.path.join(os.getcwd(),"myData")
    labelFile = 'labels.csv'
    batch_size_val = 50
    steps_per_epoch_val = 2000
    epochs_val = 10
    imageDimesions = (32, 32, 3)
    testRatio = 0.2
    validationRatio = 0.2

    current_directory = os.getcwd()
    my_data_directory = os.path.join(current_directory, 'myData')

    # Afișează calea către directorul 'myData'
    print("Calea către directorul 'myData' este:", my_data_directory)
    print("director ", current_directory)

    count = 0
    images = []
    classNo = []
    myList = os.listdir(path)
    print("Total Classes Detected:", len(myList))
    noOfClasses = len(myList)-2
    print("Importing Classes.....")
    for x in range(0, len(myList)-2):
        myPicList = os.listdir(path + "/" + str(count))
        for y in myPicList:
            curImg = cv2.imread(path + "/" + str(count) + "/" + y)
            images.append(curImg)
            classNo.append(count)
        print(count, end=" ")
        count += 1
    print(" ")
    images = np.array(images, dtype=object)
    classNo = np.array(classNo)

    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (imageDimesions[0], imageDimesions[1]))
        resized_images.append(resized_img)
    images = np.array(resized_images)

    X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

    print("Data Shapes")
    print("Train", end="")
    print(X_train.shape, y_train.shape)
    print("Validation", end="")
    print(X_validation.shape, y_validation.shape)
    print("Test", end="")
    print(X_test.shape, y_test.shape)
    assert (X_train.shape[0] == y_train.shape[
        0]), "The number of images in not equal to the number of lables in training set"
    assert (X_validation.shape[0] == y_validation.shape[
        0]), "The number of images in not equal to the number of lables in validation set"
    assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
    assert (X_train.shape[1:] == imageDimesions), " The dimesions of the Training images are wrong "
    assert (X_validation.shape[1:] == imageDimesions), " The dimesionas of the Validation images are wrong "
    assert (X_test.shape[1:] == imageDimesions), " The dimesionas of the Test images are wrong"

    data = pd.read_csv(labelFile)
    print("data shape ", data.shape, type(data))

    num_of_samples = []
    cols = 5
    num_classes = noOfClasses
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))

    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            if len(x_selected) > i:
                axs[j][i].imshow(cv2.cvtColor(x_selected[i], cv2.COLOR_BGR2RGB))
                axs[j][i].axis("off")
                if i == 2:
                    axs[j][i].set_title(str(j) + "-" + row["Name"])
                    num_of_samples.append(len(x_selected))

    num_of_samples = [x - 1 for x in num_of_samples]
    #num_of_samples = num_of_samples - 1
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

    X_train = np.array(list(map(preprocessing, X_train)), dtype=np.float32)
    X_validation = np.array(list(map(preprocessing, X_validation)), dtype=np.float32)
    X_test = np.array(list(map(preprocessing, X_test)), dtype=np.float32)

    X_train = np.expand_dims(X_train, axis=-1)
    X_validation = np.expand_dims(X_validation, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)
    dataGen.fit(X_train)
    batches = dataGen.flow(X_train, y_train,
                           batch_size=20)
    X_batch, y_batch = next(batches)

    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()

    print(type(axs))
    print(axs.shape)

    for i in range(15):
        if i < len(X_batch):
            axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
            axs[i].axis('off')
    plt.show()

    y_train = to_categorical(y_train, noOfClasses)
    y_validation = to_categorical(y_validation, noOfClasses)
    y_test = to_categorical(y_test, noOfClasses)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    model = myModel()
    print(model.summary())
    history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                       steps_per_epoch=len(X_train) // batch_size_val,
                       epochs=epochs_val,
                       validation_data=(X_validation, y_validation),
                       callbacks=[learning_rate_reduction])

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Acurracy')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])

    model.save("..\\model_trained.h5")
   # cv2.waitKey(0)
    threshold = 0.75
    #video_path = os.path.join(video_path, video_filename)
    video_path = os.path.join(video_path,'nounou.mp4')
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        print("ok")
        #sys.exit(1)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)
    cv2.namedWindow('Video with Traffic Sign Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video with Traffic Sign Recognition', 640, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("frame read successfully")
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255
            img = cv2.resize(img, (32, 32))
            img = np.expand_dims(img, axis=0)  # Adaugă o dimensiune batch

            # Recunoaște semnele de circulație
            predictions = model.predict(img)
            classIndex = np.argmax(predictions)
            probabilityValue = np.max(predictions)

            if probabilityValue > threshold:
                label = getClassName(classIndex)
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Afișează imaginea procesată
            cv2.imshow('Video with Traffic Sign Recognition', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print("no more frame to read")
            break

    cap.release()
    cv2.destroyAllWindows()
