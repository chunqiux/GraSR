# This part is modified from https://github.com/facebookresearch/moco/blob/master/moco/builder.py, which
# is under CC-BY-NC 4.0 license. The details can be referred from MoCo_LICENCE.md.
# The paper of MoCo is available on https://arxiv.org/abs/1911.05722
# Reference: He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning.
#            Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as td


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, n_key_points, base_encoder, dim, K, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(n_key_points)
        self.encoder_k = base_encoder(n_key_points)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, ld, am):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        Notice: This function is modified to process protein structure data
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        seq_len = x.shape[1]
        len_gather = concat_all_gather(torch.tensor(seq_len, dtype=torch.int).reshape(-1, 1).cuda())
        max_seq_len = int(len_gather.max())
        x_pad = torch.cat((x, torch.zeros(batch_size_this, max_seq_len - seq_len, x.shape[2]).cuda()), dim=1)
        am_pad_row = torch.cat((am, torch.zeros(batch_size_this, seq_len, max_seq_len - seq_len).cuda()), dim=2)
        am_pad = torch.cat((am_pad_row, torch.zeros(batch_size_this, max_seq_len - seq_len, max_seq_len).cuda()), dim=1)

        x_pad_gather = concat_all_gather(x_pad)
        ld_gather = concat_all_gather(ld)
        am_pad_gather = concat_all_gather(am_pad)
        batch_size_all = x_pad_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # broadcast to all gpus
        td.broadcast(idx_shuffle, src=0)
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        # shuffled index for this gpu
        gpu_idx = td.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_pad_gather[idx_this], ld_gather[idx_this], am_pad_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        # restored index for this gpu
        gpu_idx = td.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, args, test=False):
        """
        Notice: This function is modified to process protein structure data
        Input:
            im_q: a batch of query data
            im_k: a batch of key data
        Output:
            logits, targets
        """
        x_1, x_2, ld_1, ld_2, am_1, am_2 = args
        # compute query features
        q = self.encoder_q(x_1, ld_1, am_1)  # queries: NxC
        q = F.normalize(q, dim=1)

        if test:
            return q
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x_2, ld_2, am_2, idx_unshuffle = self._batch_shuffle_ddp(x_2, ld_2, am_2)
            k = self.encoder_k(x_2, ld_2, am_2)  # keys: NxC
            k = F.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(td.get_world_size())]
    td.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
