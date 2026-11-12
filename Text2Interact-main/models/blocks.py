from .layers import *


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 use_word_cond=False,
                 editable=False,
                 index=None,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        self.use_word_cond = use_word_cond

        self.sa_block = SelfAttention(latent_dim=latent_dim, text_latent_dim=None, num_heads=num_heads, dropout=dropout, index=index)
        self.ca_block = CrossAttention(latent_dim=latent_dim, text_latent_dim=latent_dim, num_heads=num_heads, dropout=dropout, index=index)
        if self.use_word_cond:
            self.text_ca_block = CrossAttention(latent_dim=latent_dim, text_latent_dim=latent_dim, num_heads=num_heads, dropout=dropout, index=index)

        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, text_emb=None, key_padding_mask=None):
        if self.use_word_cond:
            temp = self.text_ca_block(x, text_emb, emb, key_padding_mask=None)  # text has no padding
            x = x + temp
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out
