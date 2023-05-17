import numpy as np
import scipy as sp
#storing the image and the world coordinates in a list
image_points = [[757, 213], [758, 415], [758, 686], [759, 966], [1190, 172], [329, 1041], [1204, 850], [340, 159]]

world_points = [[0, 0, 0], [0, 3, 0], [0, 7, 0], [0, 11, 0], [7, 1 , 0], [0, 11, 7], [7, 9, 0], [0, 1, 7]]

comp_matrix = np.zeros([0, 12])
#using for loop to append all the 8 sets of points in comp_matrix and finding the AP=0 matrix
for i in range(len(image_points)):
    U = image_points[i][0]
    V = image_points[i][1]
    W = 1
    X = world_points[i][0]
    Y = world_points[i][1]
    Z = world_points[i][2]

    mat = [[0, 0, 0, 0, -W*X, -W*Y, -W*Z, -W, V*X, V*Y, V*Z, V],
           [W*X, W*Y, W*Z, W, 0, 0, 0, 0, -U*X, -U*Y, -U*Z, -U],
           [-V*X, -V*Y, -V*Z, -V, U*X, U*Y, U*Z, U, 0, 0, 0, 0]]
    
    mat_row_1 = mat[0]
    mat_row_2 = mat[1]
    mat_row_3 = mat[2]
    comp_matrix = np.vstack((comp_matrix, mat_row_1,mat_row_2, mat_row_3))
#finding sigular value decomposition of the comp_matrix to obtain P_Matrix
u, s, vh = np.linalg.svd(comp_matrix)
P_matrix = vh[-1]


P_matrix_updated = np.reshape(P_matrix, (3,4))
updated_P_matrix = P_matrix_updated/P_matrix_updated[-1, -1]
print("The P matrix is:\n" , updated_P_matrix)


P1, P2, P3 = np.linalg.svd(P_matrix_updated)
trans_matrix = P3[-1]
#Finding C_matrix using the same procedure used for finding the P_matrix
C_matrix = np.reshape(trans_matrix, (1,4))
updated_C_matrix = C_matrix/C_matrix[0][3]
print("The C matrix is:\n", updated_C_matrix)
#Appending identity matrix to the C_matrix
B =[[1, 0, 0, -updated_C_matrix[0][0]],
    [0, 1, 0, -updated_C_matrix[0][1]],
    [0, 0, 1, -updated_C_matrix[0][2]]]

#finding M matrix which by finding the inverse of B and multiplying it with P matrix
M_matrix = updated_P_matrix@np.linalg.pinv(B)

#Performing RQ decomposition to obtaion K and R values from M_matrix
K, R = sp.linalg.rq(M_matrix)
print("The R matrix is:\n", R)
print("The K matrix is:\n", K)
add_one = np.ones([8,1])
updated_world_list = np.hstack((world_points, add_one))

inter_mat = updated_P_matrix@np.transpose(updated_world_list)
inter_mat_1 = inter_mat/inter_mat[2]

inter_mat_2 = inter_mat_1[:-1]
#subtracting the image points with that of the modified P matrix to see if the error is below threshold
error = np.linalg.norm(image_points-inter_mat_2.T, axis=1)
for i in error:
    print("project error for image: " ,i)

