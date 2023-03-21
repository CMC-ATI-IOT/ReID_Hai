import torch.nn.functional as F
import torch.nn as nn
import torch

__all__ = ['Circle_Loss', 'CE_Loss', 'Focal_Loss', 'Triplet_Loss']

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist

class NllLoss(nn.Module):
    def __init__(self,cfgs):
        super(NllLoss,self).__init__()
        self.cfgs = cfgs
    def forward(self,output,target):
        loss = F.nll_loss(output, target)
        return loss

class Circle_Loss(nn.Module):
    def __init__(self, cfgs):
        super(Circle_Loss, self).__init__()
        self.cfgs = cfgs
    def pairwise_circleloss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
        delta_p = 1 - margin
        delta_n = margin

        logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss


    def pairwise_cosface(
            embedding: torch.Tensor,
            targets: torch.Tensor,
            margin: float,
            gamma: float, ) -> torch.Tensor:
        # Normalize embedding features
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
        logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

    def forward(self, output, target):
        loss = pairwise_cosface(output, target)
        return loss



class CE_Loss(nn.Module):
    def __init__(self, cfgs):
        super(CE_Loss, self).__init__()
        self.cfgs = cfgs

    def cross_entropy_loss(pred_class_outputs, gt_classes, eps=0.1, alpha=0.2):
        num_classes = pred_class_outputs.size(1)

        if eps >= 0:
            smooth_param = eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(pred_class_outputs, dim=1)
            smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_outputs, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return loss

    def forward(self, output, target):
        loss = cross_entropy_loss(output, target)
        return loss

class Focal_Loss(nn.Module):
    def __init__(self, cfgs):
        super(Focal_Loss, self).__init__()
        self.cfgs = cfgs

    def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'mean') -> torch.Tensor:
    # r
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))

        if not len(input.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                            .format(input.shape))

        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                            .format(input.size(0), target.size(0)))

        n = input.size(0)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))

        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1., gamma)

        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                    .format(reduction))
        return loss
    
    def forward(self, output, target):
        loss = focal_loss(output, target)
        return loss
    

class Triplet_Loss(nn.Module):
    def __init__(self, cfgs):
        super(Triplet_Loss, self).__init__()
        self.cfgs = cfgs
    
    def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


    def hard_example_mining(dist_mat, is_pos, is_neg):
        assert len(dist_mat.size()) == 2

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N]
        dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N]
        dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

        return dist_ap, dist_an


    def weighted_example_mining(dist_mat, is_pos, is_neg):
        assert len(dist_mat.size()) == 2

        is_pos = is_pos
        is_neg = is_neg
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)

        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)

        return dist_ap, dist_an


    def triplet_loss(embedding, targets, margin=0.3, norm_feat=False, hard_mining=True):
        if norm_feat:
            dist_mat = cosine_dist(embedding, embedding)
        else:
            dist_mat = euclidean_dist(embedding, embedding)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss

    def forward(self, output, target):
        loss = triplet_loss(output, target)
        return loss
    



    


