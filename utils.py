import torch

from tqdm import tqdm


def chunked_sparse_mm(A, B, chunk_size=2):
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions for multiplication."

    # Function to create a sparse tensor for a chunk of A
    def sparse_chunk(tensor, start, end, dim):
        indices = (tensor._indices()[dim] >= start) & (tensor._indices()[dim] < end)
        chunk_indices = tensor._indices()[:, indices]
        chunk_indices[dim] -= start
        chunk_values = tensor._values()[indices]
        chunk_shape = list(tensor.shape)
        chunk_shape[dim] = end - start
        return torch.sparse_coo_tensor(chunk_indices, chunk_values, chunk_shape, device=tensor.device)

    # Initialize the result tensor using the first chunk
    start = 0
    end = min(chunk_size, A.shape[1])
    A_chunk = sparse_chunk(A, start, end, 1).to('cpu')
    B_chunk = sparse_chunk(B, start, end, 0).to('cpu')
    result = torch.matmul(A_chunk, B_chunk)

    # Accumulate the remaining chunks
    for i in tqdm(range(end, A.shape[1], chunk_size)):
        start = i
        end = min(i + chunk_size, A.shape[1])
        A_chunk = sparse_chunk(A, start, end, 1).to('cpu')
        B_chunk = sparse_chunk(B, start, end, 0).to('cpu')
        result += torch.matmul(A_chunk, B_chunk)

    return result


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = chunked_sparse_mm(torch.t(X).to(torch.float32), Y.to(torch.float32))
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t()*degree


def noise(weights, ratio=0.1):
    sampling = torch.rand(weights.shape).cuda() + torch.eye(weights.shape[0]).cuda()
    sampling = (sampling > ratio).type(torch.IntTensor).cuda()
    return weights * sampling


if __name__ == '__main__':
    tX = torch.rand(3, 8)
    print(cal_weights_via_CAN(tX, 3))
