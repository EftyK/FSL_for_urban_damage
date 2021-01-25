import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager


##### DEFINITIONS ######
metric_template = np.zeros(35)

test_accuracy = np.copy(metric_template)
val_accuracy = np.copy(metric_template)
# val_loss = np.copy(metric_template)
# val_fscore = np.copy(metric_template)


##### FILL IN THE DATA ######
# BALANCING THE WEIGHTS #
# test_accuracy = np.array([0.31, 0.25, 0.30, 0.30, 0.29, 0.29, 0.28, 0.32, 0.26, 0.28,
#                           0.28, 0.28, 0.29, 0.31, 0.27, 0.25, 0.29, 0.31, 0.28, 0.26,
#                           0.30, 0.28, 0.28, 0.30, 0.21, 0.25, 0.30, 0.28, 0.29, 0.30,
#                           0.31, 0.30, 0.31, 0.29, 0.32, 0.30, 0.31, 0.31, 0.32, 0.33,
#                           0.32, 0.33, 0.32, 0.34, 0.32, 0.33, 0.33, 0.33, 0.32, 0.33,
#                           0.33, 0.34, 0.32, 0.34, 0.34, 0.32, 0.34, 0.34, 0.35, 0.36])

# val_accuracy = np.array([0.18, 0.19, 0.35, 0.43, 0.42, 0.44, 0.50, 0.51, 0.49, 0.44, 
#                          0.48, 0.48, 0.44, 0.46, 0.49, 0.43, 0.45, 0.42, 0.39, 0.44,  
#                          0.50, 0.47, 0.49, 0.49, 0.46, 0.36, 0.41, 0.44, 0.42, 0.44,
#                          0.41, 0.43, 0.40, 0.37, 0.44, 0.41, 0.36, 0.47, 0.46, 0.53,
#                          0.37, 0.39, 0.40, 0.49, 0.45, 0.53, 0.49, 0.45, 0.54, 0.54,
#                          0.55, 0.57, 0.51, 0.48, 0.56, 0.47, 0.54, 0.58, 0.59, 0.58])

# UNDERSAMPLING #


# OVERSAMPLING #
test_accuracy = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.27, 0.25, 0.26,
                          0.25, 0.31, 0.29, 0.31, 0.29, 0.34, 0.36, 0.34, 0.39, 0.38,
                          0.25, 0.39, 0.30, 0.44, 0.40, 0.42, 0.29, 0.40, 0.45, 0.44,
                          0.43, 0.42, 0.44, 0.47])

val_accuracy = np.array([0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67,
                         0.67, 0.68, 0.68, 0.68, 0.69, 0.71, 0.70, 0.72, 0.73, 0.73,
                         0.71, 0.72, 0.74, 0.75, 0.73, 0.73, 0.75, 0.73, 0.75, 0.74,
                         0.75, 0.73, 0.76, 0.77])

# val_loss = np.array([5.28, 4.99, 2.17, 2.12])
# val_fscore = np.array([0.56, 0.55, 0.48, 0.50])

detect_all_classes = 33

plt.plot(np.arange(1, len(test_accuracy) + 1), test_accuracy, 'r', label='Test') 
plt.plot(np.arange(1, len(val_accuracy) + 1), val_accuracy, 'b', label='Training')

# plt.plot(detect_all_classes, test_accuracy[detect_all_classes], 
#         detect_all_classes, val_accuracy[detect_all_classes], 
#         ls='', marker='o', label='Starts detecting all classes')

plt.axvline(x=detect_all_classes, color='k', linestyle='--', label='Model detects all classes')

plt.legend(loc="upper left")
plt.xlabel('Number of epochs') # fontsize=16, fontype='bla bla bla'
plt.ylabel('Accuracy')
plt.show()