import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import sklearn.metrics

# https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(dpi=600)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def evaluate_model(model, dataset, num_labels=14, custom_labels=None, draw_cm=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    true_class = []
    predicted_class = []

    correct = 0
    total = 0

    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))

    with torch.no_grad():
        model.eval()
        for images, labels in dataset:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            true_class.extend(labels.tolist())
            predicted_class.extend(predicted.cpu().tolist())

            c = (predicted.cpu() == labels).squeeze()
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i]
#                 print(len(labels), len(class_correct))
                class_correct[label] += c[i].item()
                class_total[label] += 1


    true_class = np.array(true_class)
    predicted_class = np.array(predicted_class)

    cm = sklearn.metrics.confusion_matrix(true_class, predicted_class)

    label_names = [
        'speed_limit', 'goods_vehicles', 'no_overtaking',
        'no_stopping', 'no_parking', 'stop', 'bicycle', 
        'hump', 'no_left', 'no_right', 'priority_to',
        'no_entry', 'yield', 'parking'    
    ]
    
    if custom_labels is not None:
        label_names = custom_labels
    
    if draw_cm:
        plot_confusion_matrix(cm, label_names, normalize=False)

    print('Accuracy of the network on the test images: %.3f %%' % (
      100 * correct / total))

    for i in range(num_labels):
        if(class_total[i] > 0):
            print('Accuracy of %5s (%d) : %.3f %%' % (
              str(i), class_total[i], 100 * class_correct[i] / class_total[i] ))
        else:
             print('Accuracy of %5s (%d): N/A' % (
              str(i), class_total[i]))
                

                
                
def model_acc(model, dataset, num_labels=14, custom_labels=None, draw_cm=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    true_class = []
    predicted_class = []

    correct = 0
    total = 0

    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))

    with torch.no_grad():
        model.eval()
        for images, labels in dataset:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            true_class.extend(labels.tolist())
            predicted_class.extend(predicted.cpu().tolist())

            c = (predicted.cpu() == labels).squeeze()
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i]
#                 print(len(labels), len(class_correct))
                class_correct[label] += c[i].item()
                class_total[label] += 1
    if total > 0:
        return correct / total
    else:
        return 0