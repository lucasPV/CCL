from __future__ import print_function

import math
import torch
import torch.nn as nn
import numpy as np


class CCL_Loss(nn.Module):
    '''
    Contextual Contrastive Loss (CCL)
    '''
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CCL_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def neighbors_features(self, indices, rks, all_features, top_k=10, duplicate=True, device="cuda"):
        """
        Computes the feature vectors of the top-k neighbors for a given set of indices based on a provided ranking.

        Parameters:
        - indices (torch.Tensor): Tensor containing the indices of the elements for which neighbors' features need to be retrieved.
        - rks (torch.Tensor): A tensor representing the ranking of all elements. This tensor determines the order in which neighbors are selected.
        - all_features (torch.Tensor): A tensor containing the feature vectors of all elements in the dataset.
        - top_k (int, optional): The number of top-ranked neighbors to retrieve for each index in 'indices'. Default is 10.
        - duplicate (bool, optional): If True, allows for the same element to appear twice. Required because each batch contains an augmented example of the same element.
        - device (str, optional): The device (e.g., "cuda" or "cpu") on which to perform computations. Default is "cuda".

        Returns:
        - torch.Tensor: A tensor containing the feature vectors of the top-k neighbors for each input index in 'indices'.
        """
        # get indices of neighbors
        neighbors_indices = []
        for k in range(top_k):
            neighbors = []
            for ind in indices:
                neighbors.append(rks[ind][k])
            neighbors_indices.append(neighbors)
        # get neighbors matrices
        neighbors_matrices = []
        for index_list in neighbors_indices:
            new_mat = torch.index_select(all_features, 0, torch.tensor(index_list).to(device))
            if duplicate:
                new_mat = torch.cat([new_mat, new_mat], dim=0)
            neighbors_matrices.append(new_mat)
        return neighbors_matrices

    def forward(self, features, labels=None, mask=None, indices=None,
                saved_features=None, saved_labels=None, saved_rks=None, epoch=0, k_start=50, total_epochs=100):
        """
        Executes the forward pass.

        Parameters:
        - features (torch.Tensor): The input feature vectors for the current batch. Hidden vector of shape [bsz, n_views, ...].
        - labels (torch.Tensor, optional): The corresponding labels for the input features. Default is None.
        - mask (torch.Tensor, optional): Contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        - indices (torch.Tensor, optional): Indices of the current batch elements. Default is None.
        - saved_features (torch.Tensor, optional): Previously saved feature vectors. Default is None.
        - saved_labels (torch.Tensor, optional): Previously saved labels corresponding to the saved features. Default is None.
        - saved_rks (torch.Tensor, optional): Previously saved ranking information for the features. Default is None.
        - epoch (int, optional): The current training epoch. Default is 0.
        - k_start (int, optional): The initial number of neighbors to consider in certain operations, adjusted based on the epoch. Default is 50.
        - total_epochs (int, optional): The total number of epochs for training. Used to adjust parameters dynamically during training. Default is 100.

        Returns:
        - A loss scalar.
        """
        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits for the original pairs
        anchor_dot_contrast_original = 1 + 1/(1+torch.norm(anchor_feature.unsqueeze(1) - contrast_feature.unsqueeze(0), dim=2))
        anchor_dot_contrast_original = anchor_dot_contrast_original
        anchor_dot_contrast_original = torch.div(anchor_dot_contrast_original, self.temperature)
        # compute logits for top-k of rks
        accumulated_matrix = torch.zeros_like(anchor_dot_contrast_original)
        accumulated_matrix_sym = torch.zeros_like(anchor_dot_contrast_original)

        # adaptive k based on log function
        if (epoch > 0):
            p_k = k_start
            p_total_epochs = total_epochs
            top_k = int((1 - math.log(epoch, p_total_epochs)) * p_k)
            if top_k <= 1:
                top_k = 1
        else:
            top_k = 15

        # compare the anchor with other elements and to the same process up to k neighbors (equation 2)
        matrices = self.neighbors_features(indices, saved_rks, saved_features, top_k=top_k, device=device)
        div = 0
        p_const = 1
        for i, mat in enumerate(matrices):
            anchor_dot_contrast = torch.nn.functional.cosine_similarity(anchor_feature.unsqueeze(1), mat.unsqueeze(0), dim=2)
            anchor_dot_contrast = 1 + 1/(1+torch.norm(anchor_feature.unsqueeze(1) - mat.unsqueeze(0), dim=2))
            anchor_dot_contrast = anchor_dot_contrast
            anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
            anchor_dot_contrast_sym = anchor_dot_contrast.clone().t()
            accumulated_matrix += anchor_dot_contrast * ((p_const)**i)
            accumulated_matrix_sym += anchor_dot_contrast_sym * ((p_const)**i)
            div += ((p_const)**i)
        # compute it as the length of a vector (equation 4)
        accumulated_matrix = accumulated_matrix / div
        accumulated_matrix = accumulated_matrix ** 2
        accumulated_matrix_sym = accumulated_matrix_sym / div
        accumulated_matrix_sym = accumulated_matrix_sym ** 2
        anchor_dot_contrast_original = (anchor_dot_contrast_original) ** 2
        summed_tensor = accumulated_matrix + accumulated_matrix_sym + anchor_dot_contrast_original
        sqrt_tensor = torch.sqrt(summed_tensor)
        anchor_dot_contrast = sqrt_tensor
        if top_k == 1:
            anchor_dot_contrast = anchor_dot_contrast_original

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
