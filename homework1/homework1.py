import csv
import numpy as np
from collections import Counter

f = open('homework1_KNN_data.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
trainData = list(rdr)[1:]
f.close()

trainLabel = np.array(trainData)[:,0]
trainLabel = trainLabel.astype(np.int32)
trainVector = np.array(trainData)[:,1:]
trainVector = trainVector.astype(np.float64)

# print(trainLabel.shape)
# print(trainVector.shape)

def predict(X, k):
    processArr1 = trainVector - X
    processArr2 = processArr1**2
    processArr3 = np.sum(processArr2, axis = 1)
    processArr4 = processArr3.argsort()[0:k]
    labelList = []
    for val in processArr4:
        labelList.append(trainLabel[val])
    result = Counter(labelList).most_common()[0][0]

    # print('P1---------------------------------------------')
    # print(processArr1)
    # print('P2---------------------------------------------')
    # print(processArr2)
    # print('P3---------------------------------------------')
    # print(processArr3)
    # print('P4---------------------------------------------')
    # print(processArr4)
    # print('P5---------------------------------------------')
    # print(labelList)
    # print('R---------------------------------------------')
    # print(result)

    return result #result is label

d1 = [1.4, 0.2]
d2 = [1.4, 0.5]
d3 = [0.9, 4.0]
d4 = [-0.1, 3.0]
d5 = [2.5, 0.1]

print()
print("1) K=3 일 때, d1 ~ d5의 각 데이터 포인트의 예상되는 label를 구하시오.")
print("[k=3 & d1] label is ", predict(d1, 3))
print("[k=3 & d2] label is ", predict(d2, 3))
print("[k=3 & d3] label is ", predict(d3, 3))
print("[k=3 & d4] label is ", predict(d4, 3))
print("[k=3 & d5] label is ", predict(d5, 3))
print()
print("2) K=5 일 때, d1 ~ d5의 각 데이터 포인트의 예상되는 label를 구하시오.")
print("[k=5 & d1] label is ", predict(d1, 5))
print("[k=5 & d2] label is ", predict(d2, 5))
print("[k=5 & d3] label is ", predict(d3, 5))
print("[k=5 & d4] label is ", predict(d4, 5))
print("[k=5 & d5] label is ", predict(d5, 5))
print()