from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


#-------------
class CausalSelfAttention(nn.Module):
    #Multi-headed attention 

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # batch of K,Q,V projections for all heads

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1 #sets a flag for this module as it needs a different initialization 
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B,T,C = x.size()
        
        qkv =self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, 2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        
        #compute affinities/calculate attention
        att = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C) #all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    #Linear layer + Non-linearity
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh') #gaussion error linear units. using approximation to reproduce gpt-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1 #flags as in attention

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    #Transformer block: communication(attention) followed by computation (ffwd)

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #normalizing and implementing residual connections 
        x = x + self.mlp(self.ln_2(x)) #normalizing and implementing residual connections
        return x
        


@dataclass 
class GPTConfig:
    block_size: int = 1024 #max seq length
    vocab_size: int = 50257 #50000 bpe merger + 256 byte tokens + end token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme. We want the embedding matrix to be shared with the linear layer in the end of the transformer ("embedding to be symetrical to disembedding" as I get it)
        self.transformer.wte.weight = self.lm_head.weight #redirects wte.weight to lm_head (points to lm_head) / weights are shared. Reduces parameters substantially 

        #parameter initialization
        self.apply(self.__init_weights) #iterates all submodules and applies init_weights

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 #as in openai implentation. Can be justified as 0.02 close to typical 1/sqrt(dmodel)
            if hasattr(module, 'SCALE_INIT'):#if it is flagged. Projections to residual layer are flagged and initialized differently
                std *= (2*self.config.n_layer)**(-0.5) # every layer has two blocks that add to residual pathway (attn and mlp), hence 2*self.n_layer is the number contributions to residual pathway
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) #samples weights from a normal distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #initializes the bias terms as 0s
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02) #same as above for embedding layers



    def forward(self, idx, targets=None):
        #idx: (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward seq of len {T}, block size is {self.config.block_size}"
        #token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #(T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) #(B,T,n_embd)
        x = tok_emb + pos_emb
        #forward blocks
        for block in self.transformer.h:
            x = block(x)
        #forward final layernorm
        x = self.transformer.ln_f(x)
        #forward classifier
        logits = self.lm_head(x) #(B,T,vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #logits flattened out (B*T, vocab_size) and targets (B*T)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

#---------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T

        #load tokens from disk and store them in memory
        with open("shakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') #generating tokenizer
        tokens = enc.encode(text) #encoding text
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        #state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T 
        buf = self.tokens[self.current_position : self.current_position+B*T+1] #creates unidimensional tensor containing what will become inputs and labels for the batch
        x = (buf[:-1]).view(B,T) #creates inputs (B,T)
        y = (buf[1:]).view(B,T) #creates labels (B,T)
        # update state / advance position in the tensor
        self.current_position += B*T
        # reset of out of bounds
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x,y

#--------------------------------------------------------------------------------------------------
#Training

#checking device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('using device: ', device)

#seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#genereating data loader
train_loader = DataLoaderLite(B=4, T=32)

#get logits
model = GPT(GPTConfig())
model.to(device)

#optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x,y = train_loader.next_batch() #loading batch
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f'step {i}, loss: {loss.item()}') #loss is a tensor with single value on the gpu, .item brings back to cpu and converts to float




import sys; sys.exit(0)
#-----------------------------------------------------------
model.eval()

num_return_sequences = 5
max_length = 30

#prefix tokens (to start generation)
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) #(T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(B,T)
x = tokens.to('cuda')

#Generation
torch.manual_seed(42) #for comparison
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    #calculate logits
    with torch.no_grad():
        logits = model(x) #(B, T, vocab_size)
        #logits of the last position
        logits = logits[:, -1 ,:] #(B, vocab_size)
        #probabilities
        probs = F.softmax(logits, dim=-1)
        #top-k sampling selects only top-50 probabilities (done by default in huggingface model)
        #helps the model not to derrail in very low prob tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # both (B, 50)
        #sample from prob distribution
        ix = torch.multinomial(topk_probs, 1) #(B,1)
        xcol = torch.gather(topk_indices, -1, ix) #(B,1)
        #append to sequence
        x = torch.cat((x, xcol), dim=1)

#print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)