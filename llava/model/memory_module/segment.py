import torch

def cal_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            #如果左侧的相似度 sim_scores[li] 大于等于当前的最大峰值 lpeak，更新 lpeak，否则停止，说明左侧的相似度已经开始下降
            else:
                break
        rpeak = sim_scores[i]
        for ri in range(i+1, n):
            if sim_scores[ri] >= rpeak:
                rpeak = sim_scores[ri]
            #如果右侧的相似度 sim_scores[ri] 大于等于当前的最大峰值 rpeak，更新 rpeak，否则停止，说明右侧的相似度已经开始下降
            else:
                break
        #当depth score的值越大，说明当前时间点相对于左右的相似度越小，即越可能是segment的边界
        depth_scores[i] = lpeak + rpeak - 2 * sim_scores[i]
    return depth_scores


def segment(features, alpha=0.5, k=None):
    # input shape: t, d
    # 对于每个时间点，计算相邻两个时间点的余弦相似度
    if features.shape[0] == 1:  # 如果只有一个时间点
        return [0]

    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_depth_score(sim_scores)

    if k is not None:
        # select by top k
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        # select by threshold (original)
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        if len(boundaries) > 15: # limit max segments to prevent from OOM: 7 comes from RMT paper
            boundaries = torch.topk(depth_scores, 15).indices.sort()[0]

    boundaries = boundaries.tolist()

    if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
        boundaries.append(features.shape[0])
    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    boundaries = sorted(set(boundaries))  # 去重并排序

    return boundaries

def adjusted_segment(features, alpha=0.5, k=None, min_distance=15, max_distance=50):
    """
    Segment a sequence of features into segments based on cosine similarity.

    Parameters:
      features: tensor of shape (t, d)
      alpha: parameter for thresholding depth scores
      k: if provided, select top-k boundaries based on depth scores
      min_distance: minimum allowed gap between consecutive boundaries
      max_distance: maximum allowed gap between consecutive boundaries;
                    if a gap is larger, extra boundaries will be inserted evenly.
    """
    # If there is only one time point, return [0] as the only boundary.
    if features.shape[0] == 1:
        return [0]

    # Compute cosine similarity between adjacent features
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_depth_score(sim_scores)

    # Determine candidate boundaries based on either top-k or thresholding
    if k is not None:
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        if len(boundaries) > 15:
            boundaries = torch.topk(depth_scores, 15).indices.sort()[0]

    boundaries = boundaries.tolist()

    # Always include the last time point as a boundary
    if not boundaries or boundaries[-1] != features.shape[0]:
        # the right boundary is not selected by the segment, so the last boundary must be len
        boundaries.append(features.shape[0])
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    print("boundaries before adjusted: ", boundaries)

    # Enforce minimum and maximum distance constraints
    # Ensure that the first and last indices are included
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)

    adjusted_boundaries = [boundaries[0]]
    for idx, b in enumerate(boundaries[1:-1], start=1):
        gap = b - adjusted_boundaries[-1]
        if gap < min_distance:
            # Skip if too close
            continue
        if gap > max_distance:
            # Compute number of extra boundaries to insert uniformly.
            X = int(gap / max_distance)
            start = adjusted_boundaries[-1]
            # Insert extra boundaries uniformly between start and b
            for i in range(1, X + 1):
                new_boundary = start + round(gap * i / (X + 1))
                # Ensure that the new boundary is strictly between start and b
                if new_boundary > adjusted_boundaries[-1] and new_boundary < b:
                    adjusted_boundaries.append(new_boundary)
        # Always add the candidate boundary.
        adjusted_boundaries.append(b)
    if boundaries[-1] not in adjusted_boundaries:
        adjusted_boundaries.append(boundaries[-1])
    print("boundaries after adjusted: ", adjusted_boundaries)

    return adjusted_boundaries

def cal_left_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            else:
                break
        depth_scores[i] = lpeak - sim_scores[i]
    return depth_scores


def segment_left(features, alpha=0.5, k=None):
    # input shape: t, d
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_left_depth_score(sim_scores)

    # print(depth_scores)

    if k is not None:
        # select by top k
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        # select by threshold (original)
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        # if len(boundaries) > 15: #limit max segments to prevent from OOM: 7 comes from RMT paper
        #     boundaries = torch.topk(depth_scores, 15).indices.sort()[0]

    boundaries = boundaries.tolist()

    # print("boudaries: ", boundaries)
    # print("features: ", features)

    # if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
    #     boundaries.append(features.shape[0]-1)
    if type(boundaries) == int or boundaries == []:
        boundaries.append(features.shape[0]-1)


    # # average segment
    # l = features.shape[0]
    # boundaires = list(range(l//(k+1)-1, l, l//(k+1)))

    # segments = []
    # index = 0
    # for bi in boundaries:
    #     segments.append(features[index: bi+1])
    #     index = bi + 1
    # if index < features.shape[0]: segments.append(features[index:])

    return boundaries

