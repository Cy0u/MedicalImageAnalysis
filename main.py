import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras 
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from guizero import App, Text, PushButton, TextBox, CheckBox
from easysettings import EasySettings

def data(dataset_path):
    images = []
    labels = []
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for image_filename in os.listdir(subfolder_path):
            if image_filename.endswith(".jpg"):
                image_path = os.path.join(subfolder_path, image_filename)
                images.append(image_path)
                labels.append(subfolder)
    df = pd.DataFrame({'image': images, 'label': labels})
    return df

def create_data_generators(train, test, val):
    image_size = (224, 224)
    batch_size = 32

    train_datagen = image.ImageDataGenerator(
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'binary')

    val_datagen= image.ImageDataGenerator(
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1)

    validation_generator = val_datagen.flow_from_directory(
        val,
        target_size = image_size,
        batch_size = batch_size,
        shuffle=True,
        class_mode = 'binary')
    
    test_datagen= image.ImageDataGenerator(
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1)
    
    test_generator = test_datagen.flow_from_directory(
        test,
        target_size = image_size,
        batch_size = batch_size,
        shuffle=True,
        class_mode = 'binary')

    return train_generator, test_generator, validation_generator

def build_model():
    class_num = 1
    global model 
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation=tf.nn.relu,input_shape=(224,224,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2)))  
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3),activation=tf.nn.relu)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2))) 
    model.add(keras.layers.Dropout(.3)) 
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3),activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2))) 
    model.add(keras.layers.Dropout(.3)) 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation=tf.nn.relu))
    model.add(keras.layers.Dropout(.3)) 
    model.add(keras.layers.Dense(128,activation=tf.nn.relu)) 
    model.add(keras.layers.Dropout(.3)) 
    model.add(keras.layers.Dense(class_num,activation=tf.nn.sigmoid))

    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

def train_model(model, train_generator, val_generator, epochs):
    lrp=ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2)
    filepath='/Users/aronpeschel/source/MedicalImageAnalysis/best_model.keras'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    call=[checkpoint,lrp]
    return model.fit(train_generator, epochs=int(epochs), validation_data=val_generator, steps_per_epoch=100, callbacks=call)

def evaluate_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator)
    print('Test Loss =', loss)
    print('Test Accuracy =', accuracy)

    y_test = test_generator.classes
    predictions = model.predict(test_generator)
    y_pred = np.where(predictions >= 0.5, 1, 0)
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    df = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})

    return y_test, y_pred

def plot_confusion_matrix(y_test, y_pred):
    class_names=['Fractured', 'Not Fractured']
    CM = confusion_matrix(y_test,y_pred)
    sns.heatmap(CM, fmt='g', center=True, cbar=False, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.title('Confusion Matrix')
    plt.show()

def plot_accuracy_graph(hist):
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Graph')
    plt.show()

def test_image():
    testImagePath = app.select_file(title="Select test image")
    if not testImagePath:
        return

    img = load_img(testImagePath, target_size=(224,224))
    imag = img_to_array(img)
    imaga = np.expand_dims(imag,axis=0) 
    ypred = model.predict(imaga)
    print(ypred)
    a=ypred[0]
    if a<0.5:
        op="Fracture"   
    else:
        op="Normal"
    fig = plt.figure()
    fig.canvas.manager.set_window_title(op)
    plt.imshow(img)

def train():
    if not trainDataPath.value or not testDataPath.value or not validateDataPath.value:
        return
    
    train_generator, test_generator, val_generator = create_data_generators(trainDataPath.value, testDataPath.value, validateDataPath.value)
    hist = train_model(model, train_generator, val_generator, iterationsValue.value)

    if(plotCheckBox.value == 1):
        plot_accuracy_graph(hist)

    evaluate(test_generator)

def evaluate(test_generator):
    y_test, y_pred = evaluate_model(model, test_generator)

    if(plotCheckBox.value == 1):
        plot_confusion_matrix(y_test, y_pred)

def selectTrainData():
    trainDataPath.value = app.select_folder(title="Select training data folder", folder=".")
    if trainDataPath.value:
        settings.setsave("trainPath", trainDataPath.value)

def selectTestData():
    testDataPath.value = app.select_folder(title="Select test data folder", folder=".")
    if testDataPath.value:
        settings.setsave("testPath", testDataPath.value)

def selectValidateData():
    validateDataPath.value = app.select_folder(title="Select validate data folder", folder=".")
    if validateDataPath.value:
        settings.setsave("validatePath", validateDataPath.value)

def iterationsChanged():
    settings.setsave("iterations", iterationsValue.value)

if __name__ == "__main__":
    build_model()

    settings = EasySettings("myconfigfile.conf")
    app = App(title="Fracture detection")
    iterations = Text(app, text="Count of training iterations", enabled=True, color="Black", width="fill")
    iterationsValue = TextBox(app, settings.get("iterations"), command=iterationsChanged)
    plotCheckBox = CheckBox(app, "Plot results")
    trainButton = PushButton(app, text="Train", command=train)
    selectTrainData = PushButton(app, text="Select train data", command=selectTrainData)
    trainDataPath = TextBox(app, settings.get("trainPath"), enabled=False, width="fill")
    selectTestData = PushButton(app, text="Select test data", command=selectTestData)
    testDataPath = TextBox(app, settings.get("testPath"), enabled=False, width="fill")
    selectValidateData = PushButton(app, text="Select validate data", command=selectValidateData)
    validateDataPath = TextBox(app, settings.get("validatePath"), enabled=False, width="fill")
    testButton = PushButton(app, text="Test image", command=test_image)

    app.display()