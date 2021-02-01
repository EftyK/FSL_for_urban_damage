import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
import matplotlib.patches as patches

##### DEFINITIONS ######
color_palette = ['#32a852', '#3289a8', '#8b57cf', '#b5319b']

def plot_row(row, data):
    rects = row.bar(models, data, color=color_palette) # set different colors for each model
    # row.set_xticklabels(models, rotation=70, fontname='Rekha')
    row.get_xaxis().set_visible(False)
    row.set_ylim([0,1])
    for rect in rects:
        height = rect.get_height()
        row.text(rect.get_x() + rect.get_width() / 2, height,'{}'.format(height), ha='center', va='bottom', fontname='Rekha', fontsize=16) # weight='bold', 
        row.axes.get_yaxis().set_ticks([])
    return rects

models = ['balance weights', 'undersampling', 'oversampling', 'protonets']
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
            [0.35, 0.53, 0.49, 0.60], # no-damage
            [0.31, 0.40, 0.41, 0.55], # minor-damage,
            [0.51, 0.68, 0.68, 0.59], # major-damage,
            [0, 0.65, 0.81, 0.83], # destroyed
            [0.29, 0.56, 0.60, 0.64] # average
            ])

recall = np.array([
        [0.13, 0.23, 0.31, 0.69], # no-damage
        [0.85, 0.79, 0.71, 0.52], # minor-damage,
        [0.44, 0.23, 0.52, 0.57], # major-damage,
        [0, 0.80, 0.72, 0.75], # destroyed
        [0.36, 0.51, 0.56, 0.64]]) # average


f_score = np.array([
        [0.19, 0.32, 0.38, 0.64], # no-damage
        [0.45, 0.53, 0.52, 0.54], # minor-damage,
        [0.47, 0.34, 0.59, 0.58], # major-damage,
        [0, 0.72, 0.76, 0.79], # destroyed
        [0.28, 0.48, 0.56, 0.64]]) # average

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
    ax.set_title(col, fontname='Rekha', y=1.1, fontsize=15)

for ax, row in zip(axes[:,0], classes):
    ax.set_ylabel(row, rotation=90, fontname='Rekha', fontsize=14)
    # ax.yaxis.set_label_coords(-0.1,0.1)


##### DEFINE LEGEND ######

rect1 = patches.Rectangle((0,0),1,1, color='#32a852', label='Balance weights')
rect2 = patches.Rectangle((0,0),1,1, color='#3289a8', label='Undersampling')
rect3 = patches.Rectangle((0,0),1,1, color='#8b57cf', label='Oversampling')
rect4 = patches.Rectangle((0,0),1,1, color='#b5319b', label='ProtoNet')


fig.legend(handles=[rect1, rect2, rect3, rect4], loc="upper center", prop={'family':'Rekha', 'size': 14, 'weight': 'bold'}, ncol=len(models))

plt.tight_layout()
plt.show()