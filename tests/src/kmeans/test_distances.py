from kmeans.distances import euclidian_dist
import numpy as np

def test_euclidian_dist_returns_correct_value():
    #when
    vector_1 = np.array([1,5,6])
    vector_2 = np.array([1,1,3])
    expected_distance = 5
    #given
    distance = euclidian_dist(vector_1,vector_2)
    #then
    np.testing.assert_allclose(distance, expected_distance)
