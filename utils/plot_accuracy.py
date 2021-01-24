import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager


##### DEFINITIONS ######
metric_template = np.zeros(28)

test_accuracy = np.copy(metric_template)
val_accuracy = np.copy(metric_template)
# val_loss = np.copy(metric_template)
# val_fscore = np.copy(metric_template)


##### FILL IN THE DATA ######
test_accuracy = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.27, 0.25, 0.26,
                          0.25, 0.31, 0.29, 0.31, 0.29, 0.34, 0.36, 0.34, 0.39, 0.38,
                          0.25, 0.39, 0.30, 0.44, 0.40, 0.42, 0.29, 0.40])

val_accuracy = np.array([0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67,
                         0.67, 0.68, 0.68, 0.68, 0.69, 0.71, 0.70, 0.72, 0.73, 0.73,
                         0.71, 0.72, 0.74, 0.75, 0.73, 0.73, 0.75, 0.73])

# val_loss = np.array([5.28, 4.99, 2.17, 2.12])
# val_fscore = np.array([0.56, 0.55, 0.48, 0.50])


plt.plot(np.arange(len(test_accuracy)), test_accuracy, 'r', label='Test') 
plt.plot(np.arange(len(val_accuracy)), val_accuracy, 'b', label='Validation')

plt.legend(loc="upper left")
plt.xlabel('Number of epochs') # fontsize=16, fontype='bla bla bla'
plt.ylabel('Accuracy')
plt.show()