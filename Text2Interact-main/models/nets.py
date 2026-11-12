import torch

from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
import random


class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size=None, min_length=15, max_length=300, dropout=0.3):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(dropout),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(dropout),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, 1)  # Regression output
        )
        self.sigmoid = nn.Sigmoid()
        self.min_length = min_length
        self.max_length = max_length
        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        raw_output = self.output(text_emb)
        output = self.sigmoid(raw_output) * (self.max_length - self.min_length) + self.min_length
        return output

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch
    
class MotionDiscriminator(nn.Module):
    """
    A binary classifier:
           MotionEncoder  →  512‑d embedding  →  1‑logit head.
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MotionEncoder(cfg)
        self.head    = nn.Linear(512, 1)

    def forward(self, batch):
        """
        batch['motions'] : (B, 2, L, D)
        batch['motion_lens'] : (B, 2)
        """
        motions = batch["motions"]                   # (B, 2, L, D)
        B, P, L, D = motions.shape                  # P == 2 (real, fake)

        print(B, P, L, D)

        motions = motions.view(B * P, L, D)          # (B*2, L, D)
        lens    = batch["motion_lens"].view(-1)      # (B*2,)

        mask = self._make_mask(L, lens)              # (B*2, L) ─ boolean
        batch_enc = {"motions": motions, "mask": mask}

        batch_enc = self.encoder(batch_enc)
        emb   = batch_enc["motion_emb"]              # (B*2, 512)
        logits = self.head(emb).squeeze(-1)          # (B*2,)

        return logits

    # --------------------------------------------------------------------- #
    @staticmethod
    def _make_mask(max_len: int, lens: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask with True for valid timesteps.
            lens : (N,) ‑‑ each element is an int length ≤ max_len
        """
        device = lens.device
        steps  = torch.arange(max_len, device=device).unsqueeze(0)      # (1,L)
        mask   = steps < lens.unsqueeze(1)                              # (N,L)
        return mask


class InterDenoiser(nn.Module):
    def __init__(self,
                input_feats,
                latent_dim=512,
                num_frames=240,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                activation="gelu",
                cfg_weight=0.,
                use_word_cond=False,
                editable=False,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.use_word_cond = use_word_cond

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.editable = editable

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size, use_word_cond=use_word_cond, editable=editable, index=i))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))



    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]

        if mask is not None:
            mask = mask[...,0]

        # Modified embedding handling
        if self.use_word_cond:
            # Word-level: cond shape (B,77,768)
            assert cond.dim() == 3, f"Word-level cond requires shape (B,77,768), got {cond.shape}"
            text_emb = self.text_embed(cond)  # (B,77,latent_dim)
            emb = self.embed_timestep(timesteps)  # (B,latent_dim)
        else:
            # Sentence-level: cond shape (B,768)
            assert cond.dim() == 2, f"Sentence-level cond requires shape (B,768), got {cond.shape}"
            text_emb = None
            emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb, text_emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, emb, text_emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output



class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.editable = getattr(cfg, "EDITABLE", False)
        self.sampling_strategy = sampling_strategy
        self.use_word_cond = getattr(cfg, "USE_WORD_COND", False)

        if cfg.DENOISER == "InterDenoiser":
            self.net = InterDenoiser(
                        self.nfeats,
                        self.latent_dim,
                        ff_size=self.ff_size,
                        num_layers=self.num_layers,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        activation=self.activation,
                        cfg_weight=self.cfg_weight,
                        use_word_cond=self.use_word_cond,
                        editable=self.editable,
            )
        else:
            raise NotImplementedError


        self.reaction_gen = getattr(cfg, "REACTION_GEN", False)
        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]

        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            reaction_gen=self.reaction_gen,
            dm_loss_type=getattr(cfg, "DM_LOSS_TYPE", None),
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask


    def compute_loss(self, batch):
        cond = batch["cond"]
        x_start = batch["motions"]
        B,T = batch["motions"].shape[:2]


        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)


        seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(x_start.device)

        model_kwargs = {"mask":seq_mask,
                      "cond":cond,
                      }

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs=model_kwargs,
            reaction_gen=self.reaction_gen,
            dm_loss_type=self.cfg.DM_LOSS_TYPE
        )
        return output

    def forward(self, batch):
        cond = batch["cond"]
        if self.reaction_gen and "motions" in batch:
            print("interdiffusion forward", batch["motions"].shape)
            x_start = batch["motions"]
        else:
            x_start = None
        B = cond.shape[0]
        T = batch["motion_lens"][0]

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            reaction_gen=self.reaction_gen,
            dm_loss_type=self.cfg.DM_LOSS_TYPE,
        )

        model_kwargs = {"mask":None,
                      "cond":cond,
                      }

        cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            cfg_model,
            (B, T, self.nfeats*2),
            clip_denoised=False,
            progress=False,
            model_kwargs=model_kwargs,
            x_start=x_start,)
        return {"output":output}




