import numpy as np

def covariance_matrix_creator(point, full_covariance_matrix, keypoints, center_point):
    first_covariance_matrix = full_covariance_matrix[0, :, :]
    indices = [sublist[0] for sublist in keypoints]
    ith_covariance_matrix = []
    ith_pt_covariance_matrix = []
    ith_jth_covariance_matrix = []
    for i, index in enumerate(indices):
        ith_covariance_matrix.append(full_covariance_matrix[index, :, :])
        sigma_x_xi = (point[0] - center_point[0]) * (keypoints[i][1][0] - center_point[0])
        sigma_y_xi = (point[1] - center_point[1]) * (keypoints[i][1][0] - center_point[0])
        sigma_z_xi = (point[2] - center_point[2]) * (keypoints[i][1][0] - center_point[0])
        sigma_x_yi = (point[0] - center_point[0]) * (keypoints[i][1][1] - center_point[1])
        sigma_y_yi = (point[1] - center_point[1]) * (keypoints[i][1][1] - center_point[1])
        sigma_z_yi = (point[2] - center_point[2]) * (keypoints[i][1][1] - center_point[1])
        sigma_x_zi = (point[0] - center_point[0]) * (keypoints[i][1][2] - center_point[2])
        sigma_y_zi = (point[1] - center_point[1]) * (keypoints[i][1][2] - center_point[2])
        sigma_z_zi = (point[2] - center_point[2]) * (keypoints[i][1][2] - center_point[2])
        covariance_i_pt_matrix = np.array([[sigma_x_xi, sigma_x_yi, sigma_x_zi],
                                  [sigma_y_xi, sigma_y_yi, sigma_y_zi],
                                  [sigma_z_xi, sigma_z_yi, sigma_z_zi]])
        ith_pt_covariance_matrix.append(covariance_i_pt_matrix)
        if i < len(indices) - 1:
            sigma_xi_xj = (keypoints[i][1][0] - center_point[0]) * (keypoints[i+1][1][0] - center_point[0])
            sigma_yi_xj = (keypoints[i][1][1] - center_point[1]) * (keypoints[i+1][1][0] - center_point[0])
            sigma_zi_xj = (keypoints[i][1][2] - center_point[2]) * (keypoints[i+1][1][0] - center_point[0])
            sigma_xi_yj = (keypoints[i][1][0] - center_point[0]) * (keypoints[i+1][1][1] - center_point[1])
            sigma_yi_yj = (keypoints[i][1][1] - center_point[1]) * (keypoints[i+1][1][1] - center_point[1])
            sigma_zi_yj = (keypoints[i][1][2] - center_point[2]) * (keypoints[i+1][1][1] - center_point[1])
            sigma_xi_zj = (keypoints[i][1][0] - center_point[0]) * (keypoints[i+1][1][2] - center_point[2])
            sigma_yi_zj = (keypoints[i][1][1] - center_point[1]) * (keypoints[i+1][1][2] - center_point[2])
            sigma_zi_zj = (keypoints[i][1][2] - center_point[2]) * (keypoints[i+1][1][2] - center_point[2])
            covariance_pt_i_pt_j_matrix = np.array([[sigma_xi_xj, sigma_xi_yj, sigma_xi_zj],
                                  [sigma_yi_xj, sigma_yi_yj, sigma_yi_zj],
                                  [sigma_zi_xj, sigma_zi_yj, sigma_zi_zj]])
            ith_jth_covariance_matrix.append(covariance_pt_i_pt_j_matrix)
    zero_matrix = np.zeros((first_covariance_matrix.shape[0], first_covariance_matrix.shape[1]))
    x_size = 0
    y_size = 0
    list_of_cov_matrices = [first_covariance_matrix] + ith_pt_covariance_matrix + [zero_matrix]
    for matrix in list_of_cov_matrices:
        x_size += matrix.shape[0]
        y_size += matrix.shape[1]
    covariance_matrix = np.empty((x_size, y_size))
    start_row_A = 0
    end_row_A = start_row_A + first_covariance_matrix.shape[0]
    start_col_A = 0
    end_col_A = start_col_A + first_covariance_matrix.shape[1]
    start_row_B = end_row_A
    end_row_B = start_row_B  + zero_matrix.shape[0]
    start_col_B = 0
    end_col_B = start_col_B + zero_matrix.shape[1]
    start_row_C = 0
    end_row_C = start_row_C  + zero_matrix.shape[0]
    start_col_C = end_col_A
    end_col_C = start_col_C + zero_matrix.shape[1]
    start_row_D = end_row_A
    end_row_D = start_row_D + zero_matrix.shape[0]
    start_col_D = end_col_A
    end_col_D = start_col_D + zero_matrix.shape[1]
    covariance_matrix[start_row_A: end_row_A, start_col_A: end_col_A] = first_covariance_matrix
    covariance_matrix[start_row_B: end_row_B, start_col_B: end_col_B] = zero_matrix
    covariance_matrix[start_row_C: end_row_C, start_col_C: end_col_C] = zero_matrix
    covariance_matrix[start_row_D: end_row_D, start_col_D: end_col_D] = zero_matrix
    start_row_i = end_row_B
    start_col_i = 0
    for mat in ith_pt_covariance_matrix:
        end_row_i = start_row_i + mat.shape[0]
        end_col_i = start_col_i + mat.shape[1]
        covariance_matrix[start_row_i: end_row_i, start_col_i: end_col_i] = mat
        start_row_i = end_row_i
    start_row_i = 0
    start_col_i = end_col_C
    for mat2 in ith_pt_covariance_matrix:
        end_row_i = start_row_i + mat2.shape[0]
        end_col_i = start_col_i + mat2.shape[1]
        covariance_matrix[start_row_i: end_row_i, start_col_i: end_col_i] = mat
        start_col_i = end_col_i
    for j in range(end_row_D, x_size, 3):
        start_row_j = j
        end_row_j = start_row_j + zero_matrix.shape[0]
        start_col_j = end_row_A
        end_col_j = start_col_j + zero_matrix.shape[1]
        covariance_matrix[start_row_j: end_row_j, start_col_j: end_col_j] = zero_matrix
    for k in range(end_col_D, y_size, 3):
        start_col_k = k
        end_col_k = start_col_k + zero_matrix.shape[0]
        start_row_k = end_row_A
        end_row_k = end_row_A + zero_matrix.shape[0]
        covariance_matrix[start_row_k: end_row_k, start_col_k: end_col_k] = zero_matrix
    start_row_i = end_row_D
    start_col_i = end_col_D
    for index, mat3 in enumerate(ith_covariance_matrix):
        end_row_i = start_row_i + mat3.shape[0]
        end_col_i = start_col_i + mat3.shape[1]
        covariance_matrix[start_row_i: end_row_i, start_col_i: end_col_i] = mat3
        covariance_matrix[end_row_i: end_row_i + ith_jth_covariance_matrix[index - 1].shape[0], start_col_i: end_col_i] = ith_jth_covariance_matrix[index - 1]
        start_col_i = end_col_i
    return covariance_matrix
