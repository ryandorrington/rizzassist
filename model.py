# Copied from tinygrad 
from os import getenv
import numpy as np
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import nn, GlobalCounters
from tinygrad.helpers import fetch, trange
from image_processor import load_batch
import matplotlib.pyplot as plt


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.prenorm, self.act = prenorm, act
        self.dropout = dropout

        self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

        self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

        self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
        self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

        self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
        self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

    def attn(self, x):
        # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
        query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
        attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
        return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

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
    def __init__(self, layers=12, embed_dim=192, num_heads=3):
        self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.cls = Tensor.ones(1, 1, embed_dim)
        self.pos_embedding = Tensor.ones(1, 197, embed_dim)
        self.tbs = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
              prenorm=True, act=lambda x: x.gelu())
            for i in range(layers)]
        self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
        self.head = (Tensor.uniform(embed_dim, 5), Tensor.zeros(5))  # Changed to output 5 classes

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
        return x[:, 0].linear(*self.head).softmax()  # Output shape (batch_size, 5) with softmax

    def load_from_pretrained(self):
        url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        
        dat = np.load(fetch(url))

        #for x in dat.keys():
        #  print(x, dat[x].shape, dat[x].dtype)

        self.embedding[0].assign(np.transpose(dat['embedding/kernel'], (3,2,0,1)))
        self.embedding[1].assign(dat['embedding/bias'])

        self.cls.assign(dat['cls'])

        # m.head[0].assign(dat['head/kernel'])
        # m.head[1].assign(dat['head/bias'])

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
    model = ViT(embed_dim=768, num_heads=12)
    model.load_from_pretrained()
    opt = nn.optim.AdamW(nn.state.get_parameters(model))

    # Load all batches
    X_all, Y_all, files_all = [], [], []
    for i in range(113):
        X, Y, files = load_batch("processed_data", i)
        X_all.append(X)
        Y_all.append(Y) 
        files_all.extend(files)
    
    # Combine into single arrays
    X_all = np.concatenate(X_all)
    Y_all = np.concatenate(Y_all)
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(X_all))
    indices = np.random.permutation(len(X_all))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, Y_train = Tensor(X_all[train_idx]), Tensor(Y_all[train_idx][:,0])
    X_test, Y_test = Tensor(X_all[test_idx]), Tensor(Y_all[test_idx][:,0])
    files_train = [files_all[i] for i in train_idx]
    files_test = [files_all[i] for i in test_idx]

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    print(Y_train[0].numpy())
    print(Y_test[0].numpy())
    print(files_train[0])
    print(files_test[0])

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(getenv("BS", 32), high=X_train.shape[0])
        loss = model(X_train[samples]).cross_entropy(Y_train[samples]).backward()
        opt.step()
        return loss
    
    @TinyJit
    @Tensor.test()
    def get_test_accuracy() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean()

    test_accuracy = float('nan')
    for i in (t:=trange(getenv("STEPS", 10000))):
        GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        if i%100 == 99: test_accuracy = get_test_accuracy().item()
        t.set_description(f"loss: {loss.item():6.2f} test_acc: {test_accuracy:5.2f}")

    # Save the trained model
    nn.state.safe_save(nn.state.get_state_dict(model), "trained_model.safetensor")

    # Display a random test image and model's prediction
    test_idx = np.random.randint(len(X_test))
    test_img = X_test[test_idx].numpy()
    true_class = Y_test[test_idx].numpy()
    pred_probs = model(X_test[test_idx:test_idx+1]).numpy()[0]
    pred_class = pred_probs.argmax()
    test_file = files_test[test_idx]
    
    print(f"\nPrediction for {test_file}:")
    print(f"Predicted class: {pred_class} (confidence: {pred_probs[pred_class]:.2f})")
    print(f"True class: {true_class}")
    
    # plt.figure(figsize=(8, 8))
    # plt.imshow(np.transpose(test_img, (1, 2, 0)))
    # plt.title(f'Predicted Class: {pred_class}\nTrue Class: {true_class}')
    # plt.axis('off')
    # plt.show()
