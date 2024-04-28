import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils import shuffle



data_dict = pickle.load(open('./data1.pickle', 'rb'))

# data = []
# labels = []
# print((len(data_dict['data'])))
# print((len(data_dict['labels'])))

# data = np.concatenate

# for each_list in data_dict['data']:
#     data.append(np.concatenate(each_list))

# for each_list in data_dict['labels']:
#     labels.append(np.concatenate(each_list))
# data = np.array(data_dict['data'])

pad = len(max(data_dict['data'], key=len))
data = np.array([i + [0]*(pad-len(i)) for i in data_dict['data']])

# print(len(data))
labels = np.array(data_dict['labels'])



x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = RandomForestClassifier()

print(len(x_test))
# print(y_train)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))
print(classification_report(y_predict, y_test))

f = open('model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
