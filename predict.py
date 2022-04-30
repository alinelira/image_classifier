import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


image_size = 224

def process_image(image): 
   
    image = tf.convert_to_tensor(image)
    image= tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model_path, top_k):
    
    original_image = Image.open(image_path)
    image = np.asarray(original_image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image)
    model = tf.keras.models.load_model(model_path ,custom_objects={'KerasLayer':hub.KerasLayer},compile=False )
    prob_list = model.predict(image)

    sorted_prob_list = prob_list[0].argsort()[::-1]
    top_classes = sorted_prob_list[:top_k]
    
    top_probs = []
    top_classes_names = []
    
    for i in range(top_k):
        
        index = top_classes[i]
        top_probs.append(prob_list[0][index])
        top_classes_names.append( class_names[str(index+1)] )
        top_classes[i] += 1

    return top_probs, top_classes, top_classes_names

parser = argparse.ArgumentParser()

parser.add_argument('img',  action='store', help='Input image path' ) #, default='./test_images/wild_pansy.jpg')
parser.add_argument('model', action='store', help='Input model path') # default='./trained_model.h5')
parser.add_argument('-top_k', help='Top K most likely classes', type=int, default=5 )
parser.add_argument('-names', help='Input class names path', default='./label_map.json')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)



if __name__ == '__main__':
    
    
    print('Image path:', args.img)
    print('\nModel path:', args.model)
    print('\nTop_k:', args.top_k)
    print('\nCategory names path:', args.names)
                    
    with open(args.names, 'r') as f:
        class_names = json.load(f)

    probs, classes, classes_names = predict(args.img, args.model, args.top_k) 
    
    for i in range(args.top_k):
        print('\nClass name: {}'.format(classes_names[i]))
        print('   Probability: {}'.format(probs[i]))
  

