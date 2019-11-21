import numpy as np
from copy import deepcopy
import os
from numpy import linalg as LA

totalMovies = 5#1682
totalUsers = 5#943

Y = np.zeros((totalUsers, totalMovies))
R = np.zeros((totalUsers, totalMovies))

count = 20
k = 5

os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")

def readTrainData(Y):

    with open("u2.txt") as myFile:
        for line in myFile:
            line = line.split("\t")
            user = int(line[0])-1
            movie = int(line[1])
            movie = movie-1
            rating = int(line[2])
            Y[user, movie] = rating
    print 'Y'
    print Y
    XO = deepcopy(Y)
    return XO

def initialize(XO, R, Y):


    for i in range(len(XO)):
        for j in range(len(XO[i])):
            if XO[i][j] == 0:
                row = np.mean(Y[i, :])
                #print row
                column = np.mean(Y[:, j])
                #print column
                mean_ = (row + column)/2.0
                XO[i][j] = mean_
            else:
                R[i][j] = 1

    #print 'XO matrix'
    print XO
    #print '************'
    #print 'R matrix'
    #print R
    #print '***********'

def latent_method(XO, R, Y, k):

    global count
    while count > 0:
        print count
        count = count - 1
        B = XO + (Y - np.multiply(R, XO))
        UK, VK = matrix_factorisation(B)
        XO = np.dot(UK, VK)
    #print 'final UK'
    #print UK
    #print 'final VK'
    #print VK
    #print 'final Ratings matrix '
    #print np.dot(UK, VK)
    return UK, VK

def ista(B, lamb, U):
    global k
    VO = np.zeros((totalMovies, k))
    VK = np.zeros((totalMovies, k))
    w, v = LA.eig(np.dot(U, np.transpose(U)))
    alpha = 1.01*np.amax(w)
    for loop in range(10):
        T = VO + np.transpose((1/alpha)*(np.dot(np.transpose(U), (B - np.dot(U, np.transpose(VO))))))
        for i in range(len(T)):
            for j in range(len(T[0])):
                #VK[i][j] = np.real(np.sign(T[i][j])*np.maximum(0, abs(T[i][j])-(lamb/(2*alpha))))
                VK[i][j] = lamb * np.real(np.sign(T[i][j]) * np.maximum(0, np.absolute(T[i][j]) - (lamb / (2 * alpha))))
        VO = VK
    return VK


def matrix_factorisation(B):
    count1 = 20
    global k
    global totalUsers
    lamb_reg = [0.1] * k
    lamb = 0.1
    U = np.random.randint(low=1, high=5, size=(totalUsers, k))
    while count1 > 0:
        count1 = count1 - 1
        U_trans = np.transpose(U)
        VK = np.linalg.lstsq(np.dot(U_trans, U), np.dot(np.transpose(U), B))[0] + np.transpose(ista(B, lamb, U))
        VK[VK < 0] = 0
        VK_trans = np.transpose(VK)
        R_trans = np.transpose(B)
        UK_temp = np.linalg.lstsq(np.dot(VK, VK_trans) + np.diag(lamb_reg), np.dot(VK, R_trans))[0]
        UK = np.transpose(UK_temp)
        UK[UK < 0] = 0
        U = UK
        if np.array_equal(np.dot(UK, VK), B) == True:
            break
    return UK, VK

def testing (UK, VK):

    records = 1000
    total_mae = 0.0
    print 'testing method'
    ratings = np.dot(UK, VK)
    #print 'ratings matrix'
    #print ratings
    #print '*********************'
    os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")
    with open("u1.test.txt") as myFile:
        for line in myFile:
            records = records + 1
            line = line.split("\t")
            user = int(line[0]) - 1
            movie = int(line[1])
            movie = movie - 1
            actual_rating = int(line[2])
            predicted_rating = ratings[user][movie]
            #print actual_rating, predicted_rating
            total_mae = total_mae + abs(predicted_rating - actual_rating)
    #print 'mae'
    mae = total_mae/records
    #print mae
    print 'nmae'
    print mae/4.0
XO = readTrainData(Y)
print 'done reading training data'
initialize(XO, R, Y)
print 'done initializing'
UK, VK = latent_method(XO, R, Y, k)
print 'rating matrix'
print np.dot(UK, VK)
print 'done latent method'
#testing(UK, VK)
