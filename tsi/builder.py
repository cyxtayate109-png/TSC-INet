import torch
import torch.nn as nn
import os

from .encoder import PretrainingEncoder
from moudle import *

# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


class TSI_CNet(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(TSI_CNet, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # domain level queues
        # temporal domain queue
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("i_queue", torch.randn(dim, K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, i_keys):
        N, C = t_keys.shape

        assert self.K % N == 0  # for simplicity

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K  # move pointer
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K  # move pointer
        self.s_queue_ptr[0] = s_ptr

        i_ptr = int(self.i_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
        i_ptr = (i_ptr + N) % self.K  # move pointer
        self.i_queue_ptr[0] = i_ptr

    def select_qk(self,input_tensor:torch.Tensor,joints_mask:list):
        # data normalization
        N, C, T, V, M = input_tensor.size()
        x = input_tensor.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Gets the sequence of nodes that are not masked out
        all_joint = set(range(V))
        remain_joint = list(all_joint - set(joints_mask))
        remain_joint = sorted(remain_joint)
        x = x[:,:,:,remain_joint]
        x = x.view(N, C, T, -1, M)
        return x

    def qk_masks(self,input_tensor:torch.Tensor,joints_mask:list):
        N, C, T, V, M = input_tensor.size()
        input_tensor = input_tensor.permute(3,0,1,2,4).contiguous()
        # masks = torch.ones(V, N, C, T, M)
        tensor0 = torch.zeros( N, C, T, M)
        for i in range(V):
            if i in joints_mask:
                input_tensor[i] = tensor0
        # masks[masks == 0] = -float('inf')
        # masks[masks == 1] = 0
        return input_tensor.permute(1,2,3,0,4)
        # return (masks + input_tensor).permute(1,2,3,0,4).contiguous()

    def forward(self, q_input, k_input):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """

        mask_frame = int(os.environ.get('KT', '10'))
        mask_joint = int(os.environ.get('KS', '8'))
        q_input = motion_att_temp_mask(q_input, mask_frame)
        k_input = motion_att_temp_mask(k_input, mask_frame)

        ignore_joints_q = central_spacial_mask(mask_joint)
        ignore_joints_k = central_spacial_mask(mask_joint)

        q_input = self.qk_masks(q_input, ignore_joints_q)
        k_input = self.qk_masks(k_input, ignore_joints_k)

        q_input = nn.functional.adaptive_avg_pool3d(q_input, output_size=(64, q_input.size(3), q_input.size(4)))
        k_input = nn.functional.adaptive_avg_pool3d(k_input, output_size=(64, k_input.size(3), k_input.size(4)))
        # print(f"q_input shape: {q_input.shape}, k_input shape: {k_input.shape}")

        # compute temporal domain level, spatial domain level and instance level features
        qt, qs, qi = self.encoder_q(q_input)  # queries: NxC

        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qi = nn.functional.normalize(qi, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            kt, ks, ki = self.encoder_k(k_input)  # keys: NxC

            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ki = nn.functional.normalize(ki, dim=1)

        # interactive loss

        positive_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
        positive_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
        positive_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
        positive_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)

        negative_ti = torch.einsum('nc,ck->nk', [qt, self.i_queue.clone().detach()])
        negative_si = torch.einsum('nc,ck->nk', [qs, self.i_queue.clone().detach()])
        negative_it = torch.einsum('nc,ck->nk', [qi, self.t_queue.clone().detach()])
        negative_is = torch.einsum('nc,ck->nk', [qi, self.s_queue.clone().detach()])

        logits_ti = torch.cat([positive_ti, negative_ti], dim=1)
        logits_si = torch.cat([positive_si, negative_si], dim=1)
        logits_it = torch.cat([positive_it, negative_it], dim=1)
        logits_is = torch.cat([positive_is, negative_is], dim=1)

        logits_ti /= self.T
        logits_si /= self.T
        logits_it /= self.T
        logits_is /= self.T

        labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long).cuda()
        labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long).cuda()
        labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long).cuda()
        labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(kt, ks, ki)

        self.last_pti = positive_ti.squeeze(1).detach()
        self.last_psi = positive_si.squeeze(1).detach()
        self.last_pit = positive_it.squeeze(1).detach()
        self.last_pis = positive_is.squeeze(1).detach()

        self.last_nti = negative_ti.detach().reshape(-1).detach()
        self.last_nsi = negative_si.detach().reshape(-1).detach()
        self.last_nit = negative_it.detach().reshape(-1).detach()
        self.last_nis = negative_is.detach().reshape(-1).detach()

        return logits_ti, logits_si, logits_it, logits_is, \
               labels_ti, labels_si, labels_it, labels_is,