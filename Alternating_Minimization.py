import numpy as np
import os

k = 3
count = 30
totalMovies = 5
totalUsers = 5

R = np.zeros((totalUsers, totalMovies))
U = np.random.rand(totalUsers, k)
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
    global count
    while count > 0:
        count = count - 1
        VK = np.linalg.lstsq(U, R)[0]
        VK[VK < 0] = 0
        UK_temp = np.linalg.lstsq(np.transpose(VK), np.transpose(R))[0]
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
