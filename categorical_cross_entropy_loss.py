import numpy as np
import math

# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0])*target_output[0] + 
#          math.log(softmax_output[1])*target_output[1] +
#          math.log(softmax_output[2])*target_output[2]
#         )

# print(loss)


softmax_output = np.array([[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0]])

# for s_output, c_target in zip(softmax_output, class_targets):
#     print(s_output[c_target])

# neg_log = -np.log(softmax_output[np.arange(len(softmax_output)), class_targets])
# average_loss = np.mean(neg_log)

# # Average loss 
# print(average_loss)

if len(class_targets.shape)==1:
       correct_pred = softmax_output[[range(len(softmax_output)), class_targets]]
elif len(class_targets.shape)==2:
        correct_pred = np.sum(softmax_output*class_targets, axis=1)


neg_loss = -np.log(correct_pred)
avg_loss = np.mean(neg_loss)

print(avg_loss)