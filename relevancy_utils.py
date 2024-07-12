import torch


def create_hook(attn_list):
    def hook(module, inputs, output):
        output[1].requires_grad_(True)
        output[1].retain_grad()
        attn_list.append(output[1])
        return output
    return hook

def compute_gradcam(attn, device):
    attn_map = attn.to(device).float().detach()
    attn_grad = attn.grad.to(device).float().detach()
    gradcam = attn_grad * attn_map
    return gradcam.clamp(0).mean(dim=[0,1])

def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition


def cal_qformer_relevancy(model, qformer_layers):

    self_att = model.vit_satt[-1]
    b, nh, q_len, kv_len = self_att.shape
    device = self_att.device
    R_i_i = torch.eye(kv_len, kv_len).to(device).float()
    for attn in model.vit_satt:
        gradcam = compute_gradcam(attn, device)   # (1+256, 1+256)
        R_i_i += torch.matmul(gradcam, R_i_i)
    vit_relevancy = R_i_i[1:, 1:]   # remove cls token

    cross_att = model.q_catt[-1]
    b, nh, q_len, kv_len = cross_att.shape
    device = cross_att.device
    R_q_q = torch.eye(q_len, q_len).to(device).float()
    R_q_i = torch.zeros(q_len, kv_len).to(device).float()
    q_catt_iter = iter(model.q_catt)
    for i, layer in enumerate(qformer_layers):
        gradcam = compute_gradcam(model.q_satt[i], device)
        r_q_q_add, r_q_i_add = apply_self_attention_rules(R_q_q, R_q_i, gradcam)
        R_q_q += r_q_q_add
        R_q_i += r_q_i_add
        if layer.has_cross_attention:
            attn = next(q_catt_iter)
            gradcam = compute_gradcam(attn, device)
            R_q_i += apply_mm_attention_rules(R_q_q, R_i_i, gradcam)
    
    return R_q_i


def cal_vit_relevancy(model):

    attn = model.vit_satt[-1]
    b, nh, q_len, kv_len = attn.shape
    device = attn.device
    R_i_i = torch.eye(kv_len, kv_len).to(device).float()
    for attn in model.vit_satt:
        attn_map = attn.to(device).float().detach()
        attn_grad = attn.grad.to(device).float().detach()
        gradcam = attn_grad * attn_map   # (1, nh, 1+576, 1+576)
        gradcam = gradcam.clamp(0).mean(dim=[0,1])   # (1+576, 1+576)
        R_i_i += torch.matmul(gradcam, R_i_i)

    return R_i_i[1:, 1:]   # remove cls token


def cal_llm_relevancy(model, attn_idx):

    attn = model.lm_satt[-1][:, :, :attn_idx, :attn_idx]
    b, nh, q_len, kv_len = attn.shape
    device = attn.device
    R_t_t = torch.eye(kv_len, kv_len).to(device).float()
    per_layer_r = []
    for attn in model.lm_satt:
        gradcam = compute_gradcam(attn, device)[:attn_idx, :attn_idx]
        R_t_t += torch.matmul(gradcam, R_t_t)
        rr = R_t_t - torch.eye(kv_len, kv_len).to(device).float()
        per_layer_r.append(rr[-1:])   # (1, curr_len)
    
    R_t_t_per_layer = torch.cat(per_layer_r, dim=0)   # (num_layers, curr_len)
    return R_t_t_per_layer


def cal_llm_relevancy_cached(model, attn_idx, gradcam_cache=[]):

    attn = model.lm_satt[attn_idx: attn_idx + model.num_llm_layers][-1]
    b, nh, q_len, kv_len = attn.shape

    if not gradcam_cache:
        assert q_len > 1, 'should be a square matrix for generating 1st word!'
        device = attn.device
        R_t_t = torch.eye(kv_len, kv_len).to(device).float()
        per_layer_r = []
        for attn in model.lm_satt[attn_idx: attn_idx + model.num_llm_layers]:
            gradcam = compute_gradcam(attn, device)
            R_t_t += torch.matmul(gradcam, R_t_t)
            rr = R_t_t - torch.eye(kv_len, kv_len).to(device).float()
            gradcam_cache.append(gradcam)
            per_layer_r.append(rr[-1:])   # (1, curr_len)
    else:
        assert q_len == 1, 'should be a single row matrix for generating >=2nd word!'
        device = gradcam_cache[-1].device
        per_layer_r = []
        R_t_t = torch.eye(kv_len, kv_len).to(device).float()
        for l, attn in enumerate(model.lm_satt[attn_idx: attn_idx + model.num_llm_layers]):
            len_cache, _ = gradcam_cache[l].shape   # previous gradcam
            gradcam = compute_gradcam(attn, device)
            # update gradcam cache with padding
            zero_column = torch.zeros(len_cache, 1).to(device)
            gradcam_cache[l] = torch.cat((gradcam_cache[l], zero_column), dim=1)
            gradcam_cache[l] = torch.cat((gradcam_cache[l], gradcam), dim=0)   # (curr_len, curr_len)
            gradcam = gradcam_cache[l]
            R_t_t += torch.matmul(gradcam, R_t_t)
            rr = R_t_t - torch.eye(kv_len, kv_len).to(device).float()
            per_layer_r.append(rr[-1:])   # (1, curr_len)
    
    R_t_t_per_layer = torch.cat(per_layer_r, dim=0)   # (num_layers, curr_len)
    return R_t_t_per_layer, gradcam_cache
