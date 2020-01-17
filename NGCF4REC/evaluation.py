import torch


class Evaluation():
    def __init__(self, k=20):
        self.k = k

    # def get_recall(self, k_items, lables):
    #     lables = lables.view(-1, 1).expand_as(k_items)
    #     hits = (k_items == lables).nonzero()[:, :-1].shape[0]
    #
    #     return hits / lables.shape[0]
    #
    # def get_mrr(self, k_items, lables):
    #     lables = lables.view(-1, 1).expand_as(k_items)
    #     ranks = (k_items == lables).nonzero()[:, -1] + 1.0
    #     rec_ranks = torch.reciprocal(ranks)
    #
    #     return torch.sum(rec_ranks).item() / lables.shape[0]

    def get_recall(self, indices, targets):
        """ Calculates the recall score for the given predictions and targets

        Args:
            indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
            targets (B): torch.LongTensor. actual target indices.

        Returns:
            recall (float): the recall score
        """
        targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = n_hits / targets.size(0)

        return recall

    def get_mrr(self, indices, targets):
        """ Calculates the MRR score for the given predictions and targets

        Args:
            indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
            targets (B): torch.LongTensor. actual target indices.
        Returns:
            mrr (float): the mrr score
        """
        targets = targets.view(-1, 1).expand_as(indices)
        # ranks of the targets, if it appears in your indices
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)  # reciprocal ranks
        mrr = torch.sum(rranks).data / targets.size(0)
        mrr = mrr.item()

        return mrr


    def evaluate(self, outs, lables):
        k_items = torch.topk(outs, self.k, 1)[1]
        recall = self.get_recall(k_items, lables)
        mrr = self.get_mrr(k_items, lables)

        return recall, mrr

