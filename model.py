from os import getenv
import numpy as np
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import nn, GlobalCounters
from tinygrad.helpers import fetch, trange
from image_processor import load_batch

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key   = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim)
    query, key, value = [
      x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1, 2)
      for y in [self.query, self.key, self.value]
    ]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1, 2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads*self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class ViT:
  def __init__(self, layers=12, embed_dim=768, num_heads=12):
    self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
    self.embed_dim = embed_dim
    self.cls = Tensor.ones(1, 1, embed_dim)
    self.pos_embedding = Tensor.ones(1, 197, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
                       prenorm=True, act=lambda x: x.gelu())
      for _ in range(layers)
    ]
    self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
    # Change the head to output a single float for regression
    self.head = (Tensor.uniform(embed_dim, 1), Tensor.zeros(1))

  def patch_embed(self, x):
    x = x.conv2d(*self.embedding, stride=16)
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).permute(order=(0,2,1))
    return x

  def __call__(self, x):
    ce = self.cls.add(Tensor.zeros(x.shape[0],1,1))
    pe = self.patch_embed(x)
    x = ce.cat(pe, dim=1)
    x = x.add(self.pos_embedding).sequential(self.tbs)
    x = x.layernorm().linear(*self.encoder_norm)
    # For regression, just produce a single scalar from the first token
    return x[:, 0].linear(*self.head)

  def load_from_pretrained(self):
    # same as your original code
    if self.embed_dim == 192:
      url = "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    elif self.embed_dim == 768:
      url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    else:
      raise Exception("no pretrained weights for configuration")

    dat = np.load(fetch(url))
    self.embedding[0].assign(np.transpose(dat['embedding/kernel'], (3,2,0,1)))
    self.embedding[1].assign(dat['embedding/bias'])
    self.cls.assign(dat['cls'])
    # We skip loading the old head because we replaced it with a single float head
    self.pos_embedding.assign(dat['Transformer/posembed_input/pos_embedding'])
    self.encoder_norm[0].assign(dat['Transformer/encoder_norm/scale'])
    self.encoder_norm[1].assign(dat['Transformer/encoder_norm/bias'])

    for i in range(12):
      self.tbs[i].query[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].reshape(self.embed_dim, self.embed_dim))
      self.tbs[i].query[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].reshape(self.embed_dim))
      self.tbs[i].key[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].reshape(self.embed_dim, self.embed_dim))
      self.tbs[i].key[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].reshape(self.embed_dim))
      self.tbs[i].value[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].reshape(self.embed_dim, self.embed_dim))
      self.tbs[i].value[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].reshape(self.embed_dim))
      self.tbs[i].out[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].reshape(self.embed_dim, self.embed_dim))
      self.tbs[i].out[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'].reshape(self.embed_dim))
      self.tbs[i].ff1[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'])
      self.tbs[i].ff1[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
      self.tbs[i].ff2[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'])
      self.tbs[i].ff2[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])
      self.tbs[i].ln1[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])
      self.tbs[i].ln1[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
      self.tbs[i].ln2[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])
      self.tbs[i].ln2[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])


if __name__ == "__main__":
    model = ViT()
    model.load_from_pretrained()

    # Use a smaller learning rate for fine-tuning
    opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-5, weight_decay=1e-4)

    # Load all batches
    X_all, Y_all, files_all = [], [], []
    for i in range(113):
        X, Y, files = load_batch("processed_data", i)
        X_all.append(X)
        Y_all.append(Y)
        files_all.extend(files)
    
    X_all = np.concatenate(X_all)
    # We'll use only the first column of Y for the beauty score
    Y_all = np.concatenate([y[:,0] for y in Y_all]).astype(np.float32)

    print(f"Loaded {len(X_all)} total samples.")
    print(f"X_all shape = {X_all.shape}, Y_all shape = {Y_all.shape}  (scores 1-5)")
    print("First few scores:", Y_all[:10])

    # Shuffle indices for train/test
    train_size = int(0.8 * len(X_all))
    indices = np.random.permutation(len(X_all))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, Y_train = Tensor(X_all[train_idx]), Tensor(Y_all[train_idx])
    X_test,  Y_test  = Tensor(X_all[test_idx]) , Tensor(Y_all[test_idx])
    files_train = [files_all[i] for i in train_idx]
    files_test =  [files_all[i] for i in test_idx]

    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        # random batch indices
        bs = int(getenv("BS", 8))
        samples = Tensor.randint(bs, high=X_train.shape[0])

        preds = model(X_train[samples])
        # reshape targets to match preds shape
        targets = Y_train[samples].reshape(shape=preds.shape)

        # Mean Squared Error for regression
        loss = (preds - targets).square().mean()
        loss.backward()
        opt.step()
        return loss

    @TinyJit
    @Tensor.test()
    def get_test_mse() -> Tensor:
        preds = model(X_test)
        targets = Y_test.reshape(shape=preds.shape)
        return (preds - targets).square().mean()

    for i in (t := trange(int(getenv("STEPS", 3000)))):
        GlobalCounters.reset()
        loss = train_step()
        if i % 100 == 0:
            mse_val = get_test_mse().item()
            t.set_description(f"step={i}, train_loss={loss.item():.4f}, test_mse={mse_val:.4f}")

    print("Done training! Saving model...")
    nn.state.safe_save(nn.state.get_state_dict(model), "trained_model.safetensor")
    print("Model saved to trained_model.safetensor")
