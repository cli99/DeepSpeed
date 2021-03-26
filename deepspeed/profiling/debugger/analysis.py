from smdebug.rules import Rule, invoke_rule
from smdebug.trials import create_trial
import numpy as np
import matplotlib.pyplot as plt

smdebug_dir = './output/mnist'

trial = create_trial(path=smdebug_dir)
print((trial.tensor_names()))

values = trial.tensor('CrossEntropyLoss_output_0').values()
values_eval = np.array(list(values.items()))
fig = plt.figure()
plt.plot(values_eval[:, 1])
fig.suptitle('Validation Accuracy', fontsize=20)
plt.xlabel('Intervals of sampling', fontsize=18)
plt.ylabel('Acuracy', fontsize=16)
fig.savefig('temp.jpg')
