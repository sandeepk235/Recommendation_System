
import numpy as np
import os

k = 2
totalMovies = 5
totalUsers = 5

R = np.zeros((totalUsers, totalMovies))
U = np.random.randint(low=1, high=5, size=(totalUsers, k))
count = 31
os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")

def readTrainData(R):

    with open("u2.txt") as myFile:
        for line in myFile:
            line = line.split("\t")
            user = int(line[0])-1
            movie = int(line[1])
            movie = movie-1
            rating = int(line[2])
            R[user, movie] = rating
    print 'R'
    print R



def matrix_factorisation(R, U):
    global k
    global count
    lamb_reg = [0.2] * k
    while count > 0:
        count = count - 1
        U_trans = np.transpose(U)
        VK = np.linalg.lstsq(np.dot(U_trans, U) + np.diag(lamb_reg), np.dot(np.transpose(U), R))[0]
        VK[VK < 0] = 0
        VK_trans = np.transpose(VK)
        R_trans = np.transpose(R)
        UK_temp = np.linalg.lstsq(np.dot(VK, VK_trans) + np.diag(lamb_reg), np.dot(VK, R_trans))[0]
        UK = np.transpose(UK_temp)
        UK[UK < 0] = 0
        U = UK
        if np.array_equal(np.dot(UK, VK), R) == True:
            break

    print 'printing final UK and VK'
    print UK
    print '***********'
    print VK
    print '***********'
    print 'printing final R'
    R_final = np.dot(UK, VK)
    R_final[R_final < 0] = 0
    print R_final

readTrainData(R)
matrix_factorisation(R, U)

