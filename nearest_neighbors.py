from sklearn.neighbors import KDTree


def get_nearest_neighbor(points, k=1):
    assert k >= 1, f'Number of nearest neighbors must be bigger than 1, not {k}'
    kdtree = KDTree(points, metric='euclidean')
    distances, nearest_points = kdtree.query(points, k=k+1)

    # points = ground_truth_cell_positions
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
    # distances, indices = nbrs.kneighbors(points)

    return distances[:, 1:], nearest_points[:, 1:].squeeze()


def get_nearest_neighbor_distances(points, k=1):
    distances, _ = get_nearest_neighbor(points, k+1)

    return distances
