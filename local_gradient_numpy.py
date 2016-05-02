import math
import random
import numpy
import datetime


def InitLFM(train, F):
    p = numpy.random.rand(maxN+1, F+1)/ math.sqrt(F)
    q = numpy.random.rand(maxM+1, F+1)/ math.sqrt(F)
    avg = sum_val * 1.0 / n
    row_bia = [0] * (maxN + 1)
    col_bia = [0] * (maxM + 1)
    return [p, q, avg, row_bia, col_bia]


def Predict(u, i, p, q):
    return numpy.dot(p[u, :], q.T[:, i])


def RMSE(rui, p, q, u, i, avg, row_bia, col_bia):
    return math.pow(rui - avg - row_bia[u] - col_bia[i] - Predict(u,i,p,q), 2)


def LearningLFM(train, F, n, alpha, lambd):
    [p, q, avg, row_bia, col_bia] = InitLFM(train, F)
    time = datetime.datetime.now()
    print time
    pre = 1
    for step in range(0, n):
        for u, v in train.items():
            for i, rui in v.items():
                i = int(i)
                u = int(u)
                pui = Predict(u, i, p, q)
                # print u,i,rui,pui
                eui = rui - pui - avg - row_bia[u] - col_bia[i]
                # print u,i,rui,pui,eui
                row_bia[u] += alpha * (eui - lambd * row_bia[u])
                col_bia[i] += alpha * (eui - lambd * col_bia[i])
                # print eui
                for k in range(0, F):
                    tmp = p[u,k]
                    p[u,k] += alpha * (q[i,k] * eui - lambd * p[u,k])
                    q[i,k] += alpha * (tmp * eui - lambd * q[i,k])
        cost = 0
        n = 0
        for u, v in train.items():
            for i, rui in v.items():
                n += 1
                i = int(i)
                u = int(u)
                cost += RMSE(rui, p, q, u, i, avg, row_bia, col_bia)
        e = math.sqrt(cost / n)
        print step,"\t"+"RMSE"+"\t", e
        if pre - e < 0.001:
            break
        alpha *= (0.9 + random.random() * 0.1)
        lambd *= (0.9 + random.random() * 0.1)
    print datetime.datetime.now(),(datetime.datetime.now() - time).seconds
    return [p, q]

print datetime.datetime.now()
File = open("uSmall.data", "r+")
train = dict()
n = 0
sum_val = 0
maxN = 0
maxM = 0
line=File.readline().rstrip()
while 1:
    line = File.readline().rstrip()
    if not line:
        break
    n += 1
    userid, itemid, record, _ = line.split(",")
    if int(userid) > maxN:
        maxN = int(userid)
    if int(itemid) > maxM:
        maxM = int(itemid)
    train.setdefault(userid, {})
    train[userid][itemid] = float(record)
    sum_val += float(record)
print n,maxN, maxM

[a, b] = LearningLFM(train, 10, 10, 0.02, 0.02)