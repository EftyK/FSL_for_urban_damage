import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
import matplotlib.patches as patches

##### DEFINITIONS ######
color_palette = ['#32a852', '#3289a8', '#8b57cf']

def plot_row(row, data):
    rects = row.bar(models, data, color=color_palette) # set different colors for each model
    # row.set_xticklabels(models, rotation=70, fontname='Rekha')
    row.get_xaxis().set_visible(False)
    for rect in rects:
        height = rect.get_height()
        row.text(rect.get_x() + rect.get_width() / 2, height,'{}'.format(height), ha='center', va='bottom', fontname='Rekha', weight='bold')
        row.axes.get_yaxis().set_ticks([])
    return rects

models = ['balance weights', 'undersampling', 'oversampling']
classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'average']
metrics = ['Precision', 'Recall', 'F-score']
metric_template = np.zeros((len(models), len(classes)))

precision = np.copy(metric_template)
recall = np.copy(metric_template)
f_score = np.copy(metric_template)


##### FILL IN THE DATA ######
# each row follows the format [m1, m2, m3, m4], where:
# m1: balance weights
# m2: undersampling
# m3: oversampling
# m4: protonets
precision = np.array([
            [0.35, 0.53, 0.32], # no-damage
            [0.31, 0.4, 0.24], # minor-damage,
            [0.51, 0.68, 0.64], # major-damage,
            [0, 0.65, 0], # destroyed
            [0.29, 0.56, 0.3] # average
            ])

recall = np.array([
        [0.13, 0.23, 0.89], # no-damage
        [0.85, 0.79, 0.17], # minor-damage,
        [0.44, 0.23, 0.3], # major-damage,
        [0, 0.8, 0], # destroyed
        [0.36, 0.51, 0.34]]) # average


f_score = np.array([
        [0.19, 0.32, 0.47], # no-damage
        [0.45, 0.53, 0.19], # minor-damage,
        [0.47, 0.34, 0.41], # major-damage,
        [0, 0.72, 0], # destroyed
        [0.28, 0.48, 0.27]]) # average

# m1 accuracy:  0.36
# m2 accuracy: 0.51
# m3 accuracy:  0.34


##### PLOT ######
fig, axes = plt.subplots(len(classes), len(metrics), sharex=True)

for row, i in zip(axes, range(len(classes))):
    for col  in row:
        plot_row(row[0], precision[i]) 
        plot_row(row[1], recall[i]) 
        plot_row(row[2], f_score[i])

##### ADD TITLES FOR COLUMNS AND ROWS ######
for ax, col in zip(axes[0], metrics):
    ax.set_title(col, fontname='Rekha', y=1.1)

for ax, row in zip(axes[:,0], classes):
    ax.set_ylabel(row, rotation=45, fontname='Rekha')
    ax.yaxis.set_label_coords(-0.1,0.1)


##### DEFINE LEGEND ######

rect1 = patches.Rectangle((0,0),1,1, color='#32a852', label='model1')
rect2 = patches.Rectangle((0,0),1,1, color='#3289a8', label='model2')
rect3 = patches.Rectangle((0,0),1,1, color='#8b57cf', label='model3')

fig.legend(handles=[rect1, rect2, rect3], loc="upper left")

plt.tight_layout()
plt.show()