import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """

    # # load the image
    # image = mpimg.imread(img)

    # # load the labels
    # with open('labels.txt', 'r') as f:
    #     labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # # load the model
    # interpreter = tf.lite.Interpreter(model_path='static/model/converted_model.tflite')
    # interpreter.allocate_tensors()

    # # get model input details and resize image
    # input_details = interpreter.get_input_details()
    # iw = input_details[0]['shape'][2]
    # ih = input_details[0]['shape'][1]
    # image = image.resize((iw, ih)).convert(mode='RGB')

    # # set model input and invoke
    # input_data = np.array(image).reshape((ih, iw, 3))[None, :, :, :]
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()

    # # read output and dequantize
    # output_details = interpreter.get_output_details()[0]
    # output = np.squeeze(interpreter.get_tensor(output_details['index']))
    # if output_details['dtype'] == np.uint8:
    #     scale, zero_point = output_details['quantization']
    #     output = scale * (output - zero_point)

    # # return the top label and its score
    # ordered = np.argpartition(-output, 1)
    # label_i = ordered[0]
    # result = {'label': labels[label_i], 'score': output[label_i]}

    # return result
    # Read an image from a file into a numpy array
    img = mpimg.imread(img)
    # Convert to float32
    img = tf.cast(img, tf.float32)
    # Expand img dimensions from (224, 224, 3) to (1, 224, 224, 3) for set_tensor method call
    img = np.expand_dims(img, axis=0)

    tflite_model_file = 'static/model/converted_model.tflite'

    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    prediction = []
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    prediction.append(interpreter.get_tensor(output_index))

    predicted_label = np.argmax(prediction)
    class_names = ['rock', 'paper', 'scissors']
    
    return class_names[predicted_label]



def plot_category(img):
    """Plot the input image

    Args:
        img [jpg]: image file
    """

    img = mpimg.imread(img)
    # Remove the plotting ticks
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    # matplotlib would not overwrite previous image, so using this work around
    # https://stackoverflow.com/questions/49039581/matplotlib-savefig-will-not-overwrite-old-files
    strFile = 'static/images/output.png'
    if os.path.isfile(strFile):
        os.remove(strFile)
    # Save the image with the file name that result.html is using as its img src
    plt.savefig(strFile)
