import torch
import torch.nn as nn

### Propagation Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        '''
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        '''
        B, N, D = x.size()
        x = self.linear(x.view(B * N, D))

        if residual is None:
            x = self.relu(x)
        else:
            x = self.relu(x + residual.view(B * N, self.output_size))

        return x.view(B, N, self.output_size)

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)

class PropModuleDiffDen(nn.Module):
    def __init__(self, config, use_gpu=False):

        super(PropModuleDiffDen, self).__init__()

        self.config = config
        nf_effect = config['train']['particle']['nf_effect']
        self.nf_effect = nf_effect
        self.add_delta = config['train']['particle']['add_delta']

        self.use_gpu = use_gpu

        # particle encoder
        # input: pusher movement (3), attr (1), density (1)
        self.particle_encoder = ParticleEncoder(
            3 + 1 + 1, nf_effect, nf_effect)

        # relation encoder
        # input: attr * 2 (2), state offset (3), density (1)
        self.relation_encoder = RelationEncoder(
            2 + 3 + 1, nf_effect, nf_effect)

        # input: (1) particle encode (2) particle effect, density (1)
        self.particle_propagator = Propagator(
            2 * nf_effect + 1, nf_effect)

        # input: (1) relation encode (2) sender effect (3) receiver effect, density (1)
        self.relation_propagator = Propagator(
            nf_effect + 2 * nf_effect + 1, nf_effect)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3)

    def forward(self, a_cur, s_cur, s_delta, Rr, Rs, particle_dens, verbose=False):
        # a_cur: B x particle_num -- indicating the type of the objects, slider or pusher
        # s_cur: B x particle_num x 3 -- position of the objects
        # s_delta: B x particle_num x 3 -- impulses of the objects
        # Rr: B x rel_num x particle_num
        # Rs: B x rel_num x particle_num
        # particle_dens: B
        B, N = a_cur.size()
        _, rel_num, _ = Rr.size()
        nf_effect = self.nf_effect

        particle_dens = particle_dens / 5000.

        pstep = 3

        Rr_t = Rr.transpose(1, 2) # TODO: add .continuous()? # B x particle_num x rel_num
        Rs_t = Rs.transpose(1, 2) # B x particle_num x rel_num

        # receiver_attr, sender_attr
        a_cur_r = Rr.bmm(a_cur[..., None]) # B x rel_num x 1
        a_cur_s = Rs.bmm(a_cur[..., None]) # B x rel_num x 1

        # receiver_state, sender_state
        s_cur_r = Rr.bmm(s_cur) # B x rel_num x 3
        s_cur_s = Rs.bmm(s_cur) # B x rel_num x 3

        # particle encode
        particle_encode = self.particle_encoder(
            torch.cat([s_delta, a_cur[:, :, None], particle_dens[:, None, None].repeat(1, N, 1)], 2)) # B x particle_num x nf_effect
        particle_effect = particle_encode

        # relation encode
        relation_encode = self.relation_encoder(
            torch.cat([a_cur_r, a_cur_s, s_cur_r - s_cur_s, particle_dens[:, None, None].repeat(1, rel_num, 1)], 2)) # B x rel_num x nf_effect

        for i in range(pstep):
            effect_r = Rr.bmm(particle_effect) # B x rel_num x nf_effect
            effect_s = Rs.bmm(particle_effect) # B x rel_num x nf_effect
            
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s, particle_dens[:, None, None].repeat(1, rel_num, 1)], 2)) # B x rel_num x nf_effect

            effect_rel_agg = Rr_t.bmm(effect_rel) # B x particle_num x nf_effect
            
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg, particle_dens[:, None, None].repeat(1, N, 1)], 2),
                residual=particle_effect)
        
        # B x particle_num x 3
        particle_pred = self.particle_predictor(particle_effect)

        return particle_pred + s_cur

class PropNetDiffDenModel(nn.Module):

    def __init__(self, config, use_gpu=False):
        super(PropNetDiffDenModel, self).__init__()

        self.config = config
        self.adj_thresh = config['train']['particle']['adj_thresh']
        self.model = PropModuleDiffDen(config, use_gpu)

    def predict_one_step(self, a_cur, s_cur, s_delta, particle_dens, particle_nums=None):
        # a_cur: B x particle_num
        # s_cur: B x particle_num x 3
        # s_delta: B x particle_num x 3
        # particle_nums: B
        # particle_dens: B
        assert type(a_cur) == torch.Tensor
        assert type(s_cur) == torch.Tensor
        assert type(s_delta) == torch.Tensor
        assert a_cur.shape == s_cur.shape[:2]
        assert s_cur.shape == s_delta.shape

        B, N = a_cur.size()

        # s_receiv, s_sender: B x particle_num x particle_num x 3
        s_receiv = (s_cur + s_delta)[:, :, None, :].repeat(1, 1, N, 1)
        s_sender = (s_cur + s_delta)[:, None, :, :].repeat(1, N, 1, 1)

        # dis: B x particle_num x particle_num
        # adj_matrix: B x particle_num x particle_num
        threshold = self.adj_thresh * self.adj_thresh
        dis = torch.sum((s_sender - s_receiv)**2, -1)
        max_rel = min(10, N)
        topk_res = torch.topk(dis, k=max_rel, dim=2, largest=False)
        topk_idx = topk_res.indices
        topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32, device=dis.device)
        topk_bin_mat.scatter_(2, topk_idx, 1)
        adj_matrix = ((dis - threshold) < 0).float()
        adj_matrix = adj_matrix * topk_bin_mat
        if particle_nums is not None:
            for b in range(B):
                adj_matrix[b, particle_nums[b]:, :] = 0
                adj_matrix[b, :, particle_nums[b]:] = 0
        n_rels = adj_matrix.sum(dim=(1,2))
        n_rel = n_rels.max().long().item()
        rels_idx = []
        rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
        rels_idx = torch.hstack(rels_idx).to(device=s_cur.device, dtype=torch.long)
        rels = adj_matrix.nonzero()
        Rr = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
        Rs = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
        Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
        Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
        s_pred = self.model.forward(a_cur, s_cur, s_delta, Rr, Rs, particle_dens)

        return s_pred
