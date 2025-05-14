from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, VariantAwarePatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from layers.Embed import DataEmbedding_wo_pos
from math import sqrt

transformers.logging.set_verbosity_error()

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k
        
    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        
        return x_season, x_trend
    
class VariantTemporalFusion(nn.Module):
    """
    ① variant-wise attention  → ② time-wise attention
    결과: [B, T, D]  (variant 축을 요약하되, 시간 패턴이 반영된 벡터)
    """
    def __init__(self, variants: int, d_model: int, heads: int = 4):
        super().__init__()
        self.V = variants
        self.D = d_model
        self.variant_embed = nn.ModuleList([nn.Linear(1, d_model) for _ in range(variants)])

        self.var_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

        self.time_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

        self.norm_var  = nn.LayerNorm(d_model)
        self.norm_time = nn.LayerNorm(d_model)

        self.var_pool = nn.Linear(variants * d_model, d_model)

    def forward(self, x):                 # x: [B, T, V]
        B, T, V = x.size()
        assert V == self.V

        toks = [self.variant_embed[v](x[:, :, v:v+1].to(torch.bfloat16))         # [B, T, D]
                for v in range(V)]
        x_var = torch.stack(toks, dim=2)                      # [B, T, V, D]

        x1 = x_var.view(B*T, V, self.D)
        v_attn, _ = self.var_attn(x1, x1, x1)                 # [B*T, V, D]
        v_attn = self.norm_var(x1 + v_attn).view(B, T, V, self.D)

        x2 = v_attn.permute(0, 2, 1, 3).contiguous()          # [B, V, T, D]
        x2 = x2.view(B*V, T, self.D)
        t_attn, _ = self.time_attn(x2, x2, x2)                # [B*V, T, D]
        t_attn = self.norm_time(x2 + t_attn).view(B, V, T, self.D)

        t_attn = t_attn.permute(0, 2, 1, 3).contiguous()      # [B, T, V, D]
        t_attn = t_attn.reshape(B, T, V*self.D)               
        out    = self.var_pool(t_attn)                        # [B, T, D]

        return out

class MultiTrendResidualVariantTemporalFusion(nn.Module):
    def __init__(self, variants, d_model, heads=4):
        super().__init__()
        self.trend1_fusion = VariantTemporalFusion(variants, d_model, heads)
        self.trend2_fusion = VariantTemporalFusion(variants, d_model, heads)
        self.res3_fusion   = VariantTemporalFusion(variants, d_model, heads)

        self.cross_trend  = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cross_final  = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc    = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward(self, t1, t2, r3):          # 모두 [B, T, V]
        t1 = self.trend1_fusion(t1)             # [B, T, D]
        t2 = self.trend2_fusion(t2)
        r3 = self.res3_fusion(r3)

        # ---- Trend cross-attention (t1 ← t2) --------------
        kv = t2       # [B, 2T, D]
        t_fused = self.cross_trend(t1, kv, kv)[0]
        t_fused = self.norm1(t1 + t_fused)

        kv2 = self.cross_final(t_fused, r3, r3)[0]
        out = self.norm2(t_fused + kv2)

        return self.fc(out)                     # [B, T, D]
    
class VariantAwareFusion(nn.Module):
    def __init__(self, variants, llm_dim, num_heads=4):
        super(VariantAwareFusion, self).__init__()
        
        self.variants = variants
        self.llm_dim = llm_dim
        
        self.variant_embeddings = nn.ModuleList([
            nn.Linear(1, llm_dim) for _ in range(variants)
        ])
        
        self.variant_attention = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(llm_dim)
        self.fc = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.ReLU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, x):
        B, T, V = x.shape
        
        variant_emb_list = []
        for v in range(V):
            emb = self.variant_embeddings[v](x[:, :, v].unsqueeze(-1))
            variant_emb_list.append(emb.unsqueeze(2))
            
        variant_emb = torch.cat(variant_emb_list, dim=2)
        
        variant_emb_reshaped = variant_emb.view(B*T, V, self.llm_dim)
        
        attn_output, _ = self.variant_attention(
            variant_emb_reshaped, variant_emb_reshaped, variant_emb_reshaped
        )
        
        fused_output = attn_output.mean(dim = 1)
        
        fused_output = fused_output.view(B, T, self.llm_dim)
        
        fused_output = self.norm(fused_output)
        fused_output = self.fc(fused_output)
        
        return fused_output
    
class FusionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Parameters:
        - d_model: feature dimension (e.g., 32)
        - n_heads: number of attention heads
        """
        super(FusionBlock, self).__init__()

        # Cross-attn: res ← trend
        self.attn_res_to_mean = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_res_to_res_mean = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # Cross-attn: trend ← res
        self.attn_mean_to_res = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_res_mean_to_res = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, mean_i, res_mean_i, res_res_i):
        """
        Inputs:
        - mean_i:      [B, T, D]
        - res_mean_i:  [B, T, D]
        - res_res_i:   [B, T, D]

        Returns:
        - components: list of [B, T, D] tensors [res_fused, trend_fused]
        """

        # residual ← trend (Q = residual)
        attn_res_mean, _ = self.attn_res_to_mean(res_res_i, mean_i, mean_i)
        attn_res_mid, _ = self.attn_res_to_res_mean(res_res_i, res_mean_i, res_mean_i)
        res_fused = attn_res_mean + attn_res_mid  # [B, T, D]

        # trend ← residual (Q = trend)
        attn_mean_res, _ = self.attn_mean_to_res(mean_i, res_res_i, res_res_i)
        attn_mid_res, _ = self.attn_res_mean_to_res(res_mean_i, res_res_i, res_res_i)
        trend_fused = attn_mean_res + attn_mid_res  # [B, T, D]

        return [res_fused, trend_fused]
    
    
class CrossAttentionReducer(nn.Module):
    def __init__(self,
                 llm_dim: int,
                 vocab_num: int,
                 heads: int = 8,
                 proj_dim: int | None = None):        # ⬅️ 줄이고 싶은 크기를 직접 넘길 수도
        super().__init__()
        self.heads = heads

        # ───── 작은 투사 차원 설정 ─────
        self.inner_dim = proj_dim if proj_dim is not None else llm_dim // 6
        assert self.inner_dim % heads == 0, "inner_dim must be divisible by heads"
        self.head_dim = self.inner_dim // heads
        self.scale = self.head_dim ** -0.5
        # ──────────────────────────────

        self.vocab_num = vocab_num

        # 학습 쿼리(크기는 inner_dim)
        self.learned_queries = nn.Parameter(torch.randn(vocab_num, self.inner_dim))

        # Q, K, V 투사: 입력 llm_dim → inner_dim
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(llm_dim,      self.inner_dim, bias=False)
        self.v_proj = nn.Linear(llm_dim,      self.inner_dim, bias=False)

        # out_proj: inner_dim → llm_dim
        self.out_proj = nn.Linear(self.inner_dim, llm_dim, bias=False)

    def forward(self, token_input):          # token_input: [T, llm_dim]
        T, H, D = token_input.size(0), self.heads, self.head_dim

        q = self.q_proj(self.learned_queries)          # [vocab, inner_dim]
        k = self.k_proj(token_input)                   # [T,     inner_dim]
        v = self.v_proj(token_input)                   # [T,     inner_dim]

        # H-way 분할
        q = q.reshape(self.vocab_num, H, D).transpose(0, 1)   # [H, vocab, D]
        k = k.reshape(T,              H, D).transpose(0, 1)   # [H, T,     D]
        v = v.reshape(T,              H, D).transpose(0, 1)   # [H, T,     D]

        attn = (q @ k.transpose(-2, -1)) * self.scale         # [H, vocab, T]
        attn = F.softmax(attn, dim=-1)

        out = attn @ v                                        # [H, vocab, D]
        out = out.transpose(0, 1).reshape(self.vocab_num, self.inner_dim)

        return self.out_proj(out)                             # [vocab, llm_dim]

        
        

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.variants = configs.variants
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_method = configs.down_sampling_method
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout, self.variants, self.seq_len)

        self.variant_patch_embedding = VariantAwarePatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout, self.variants, self.seq_len)

        
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    #   configs.dropout)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.mapping_layer = CrossAttentionReducer(self.word_embeddings.shape[1], self.num_tokens)
        
        # self.cross_attention = FusionBlock(configs.d_model, configs.n_heads)
        
        # self.variants_aware_fusion = MultiTrendResidualVariantFusion(self.variants, configs.d_model, 4)
        
        self.variants_temporal_fusion = MultiTrendResidualVariantTemporalFusion(
            variants=self.variants,
            d_model=configs.d_model,
            heads=4
        )
        
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.preprocess = series_decomp(configs.moving_avg)
        

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers1 = Normalize(configs.enc_in, affine=False)
        self.normalize_layers2 = Normalize(configs.enc_in, affine=False)
        self.normalize_layers3 = Normalize(configs.enc_in, affine=False)
        self.normalize_layers4 = Normalize(configs.enc_in, affine=False)
        
        
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc
    
    def pre_enc(self, x_enc):
        res, moving_mean = self.preprocess(x_enc)
        
        return res, moving_mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        res, mean = self.pre_enc(x_enc) # x_enc_decomp[0] : res, x_enc_decomp[1] : moving_mean
        res_res, res_mean = self.pre_enc(res) # x_enc_decomp_2[0] : res_res, x_enc_decomp_2[1] : res_moving_mean
        # res_res_res, res_res_mean = self.pre_enc(res_res)
        
        source_embeddings = self.mapping_layer(self.word_embeddings)
        
        x_enc = self.normalize_layers1(x_enc, 'norm')
        
        mean = self.normalize_layers2(mean, 'norm')
        res_res = self.normalize_layers4(res_res, 'norm')
        # res_res_mean = self.normalize_layers(res_res_mean, 'norm')
        res_mean = self.normalize_layers3(res_mean, 'norm')
        
        fusion_output = self.variants_temporal_fusion(mean, res_mean, res_res)
                
        
        # print(fusion_output.shape)
        # print(source_embeddings.shape)
                
        reprogrammed_fusion_output = self.reprogramming_layer(fusion_output, source_embeddings, source_embeddings)
        
        
        B, T, N = x_enc.size()
        x_enc = x_enc.contiguous()

        # x_enc: [B, T, N]
        min_values, _ = x_enc.min(dim=1)                   # [B, N]
        max_values, _ = x_enc.max(dim=1)                   # [B, N]
        median_values  = x_enc.median(dim=1).values        # [B, N]
        trends         = torch.diff(x_enc, dim=1).sum(dim=1)  # [B, N]
                
 
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str([round(num.item(), 1) for num in min_values[b]])
            max_values_str = str([round(num.item(), 1) for num in max_values[b]])
            median_values_str = str([round(num.item(), 1) for num in median_values[b]])
            trends_values_str = str(["up" if round(num.item(), 1) > 0 else "down" for num in trends[b]])
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"The number of variants for this dataset is {str(self.variants)} "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"trend {trends_values_str}, "
                f"Now, please forecast the next {self.pred_len} steps.\n"
            )
            

            prompt.append(prompt_)
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        
        enc_out, n_vars = self.variant_patch_embedding(x_enc.to(torch.bfloat16))
        
        
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        
        llama_enc_out = torch.cat([
            prompt_embeddings, 
            reprogrammed_fusion_output,  # mid 
            enc_out                      # last
        ], dim=1)
        
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state   # [B, L_total, llm_dim]
        dec_out = dec_out[:, :, :self.d_ff]                                       # [B, L_total, d_ff]

        patch_token_len = self.patch_nums * n_vars              
        dec_patch = dec_out[:, -patch_token_len:, :]             

        dec_patch = dec_patch.view(B, n_vars, self.patch_nums, self.d_ff)  
        dec_patch = dec_patch.permute(0, 1, 3, 2).contiguous()             

        dec_patch = self.output_projection(dec_patch)         
        dec_out  = dec_patch.permute(0, 2, 1).contiguous()     
        dec_out  = self.normalize_layers1(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
