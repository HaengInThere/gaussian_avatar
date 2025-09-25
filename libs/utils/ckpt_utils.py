# libs/utils/ckpt_utils.py
from typing import Dict, Optional
import os
import torch
import torch.nn as nn
import math

def save_full_checkpoint(
    save_dir: str,
    iteration: int,
    stage: str,
    head_model: torch.nn.Module,
    uv_net: torch.nn.Module,
    env_light: Optional[torch.nn.Module] = None,
    cfg: Optional[dict] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "iteration": iteration,
        "stage": stage,
        "cfg": cfg,
        "head_model": head_model.state_dict(),
        "uv_net": uv_net.state_dict(),
        "optimizers": {
            "head_model": (head_model.optimizer.state_dict() if getattr(head_model, "optimizer", None) is not None else None),
            "uv_net":     (uv_net.optimizer.state_dict()     if getattr(uv_net, "optimizer", None) is not None else None),
            "env": (
                env_light.optimizer.state_dict()
                if (env_light is not None and hasattr(env_light, "optimizer") and env_light.optimizer is not None)
                else None
            ),
        },
        "env_light": (env_light.state_dict() if env_light is not None else None),
    }
    torch.save(ckpt, os.path.join(save_dir, f"ckpt_{iteration:06d}.pt"))


def model_training_setup(args, cfg_training: dict, head_model, uv_net, env_light=None):
    train_stage = getattr(args, "train_stage", None)
    start_iter = 1
    resume_path = getattr(args, "resume_checkpoint", None)
    ckpt = None

    # 1) ckpt를 먼저 읽기만 함
    if resume_path and os.path.isfile(resume_path):
        print(f"[RESUME] Loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        start_iter = int(ckpt.get("iteration", 0)) + 1
        print(f"[RESUME] Resume at iter {start_iter} >>> stage => {train_stage}")

    # 2) 옵티마/스케줄러 생성
    head_model.training_setup(cfg_training, stage=train_stage)
    uv_net.training_setup(cfg_training, stage=train_stage)
    if env_light is not None:
        env_light.training_setup(cfg_training)

    # 3) 가중치 로드 (여기서 덮어써서 re-init을 무력화)
    if ckpt is not None:
        rep_h = force_load_model_from_ckpt(head_model, ckpt.get('head_model', {}), verbose=1)
        rep_u = force_load_model_from_ckpt(uv_net,     ckpt.get('uv_net',     {}), verbose=1)
        if ckpt['env_light'] is not None and (env_light is not None):
            rep_l = force_load_model_from_ckpt(env_light, ckpt['env_light'], verbose=1)
        # 간단한 진단
        print(f"[RESUME] uv_net loaded={len(rep_u['loaded'])}, mismatched={len(rep_u['mismatched'])}, missing={len(rep_u['missing'])}")

    # 4) 옵티마 상태 복구
    if ckpt is not None and 'optimizers' in ckpt:
        opt_sd = ckpt['optimizers']

        def _load_opt(opt, sd, module: nn.Module):
            if opt is None or sd is None:
                return
            try:
                opt.load_state_dict(sd)
                dev = next(module.parameters()).device
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(dev)
            except Exception as e:
                print(f"[RESUME][warn] optimizer state load failed: {e}")

        _load_opt(getattr(head_model, 'optimizer', None), opt_sd.get('head_model', None), head_model)
        _load_opt(getattr(uv_net,     'optimizer', None), opt_sd.get('uv_net', None),     uv_net)
        if env_light is not None and hasattr(env_light, 'optimizer'):
            _load_opt(env_light.optimizer, opt_sd.get('env', None), env_light)

    # 5) 스케줄러 정렬
    if getattr(head_model, 'scheduler', None) is not None:
        try: head_model.scheduler.last_epoch = start_iter - 1
        except Exception: pass
    if getattr(uv_net, 'scheduler', None) is not None:
        try: uv_net.scheduler.last_epoch = start_iter - 1
        except Exception: pass

    return start_iter


import torch
import torch.nn as nn

def force_load_model_from_ckpt(
    model: nn.Module,
    ckpt_sd: dict,
    *,
    strict: bool = False,
    verbose: int = 1,
    allow_partial: bool = False,
):
    model_sd = model.state_dict()

    def has_module_prefix(keys):
        return sum(1 for k in keys if k.startswith("module.")) > (len(keys) // 2)

    def normalize_keys(sd, ref_keys):
        sd_keys = list(sd.keys()); ref_keys = list(ref_keys)
        if not sd_keys or not ref_keys: return sd
        sd_has = has_module_prefix(sd_keys); ref_has = has_module_prefix(ref_keys)
        if sd_has and not ref_has:
            return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        if (not sd_has) and ref_has:
            return {(("module."+k) if not k.startswith("module.") else k): v for k, v in sd.items()}
        return sd

    ckpt_sd = normalize_keys(ckpt_sd, model_sd.keys())

    loaded, partial, missing, mismatched, unexpected, overwrited = [], [], [], [], [], []

    def parent_and_attr(root: nn.Module, dotted: str):
        parts = dotted.split("."); mod = root
        for p in parts[:-1]: mod = getattr(mod, p)
        return mod, parts[-1]

    param_names = {n for n, _ in model.named_parameters()}
    buffer_names = {n for n, _ in model.named_buffers()}

    # 대표 디바이스/타입 추정
    try:
        rep = next(model.parameters())
        default_device, default_dtype = rep.device, rep.dtype
    except StopIteration:
        default_device, default_dtype = torch.device("cpu"), torch.float32

    with torch.no_grad():
        for name, target in model_sd.items():
            if name not in ckpt_sd:
                missing.append(name); continue

            parent, attr = parent_and_attr(model, name)
            exists = getattr(parent, attr, None)

            # 현재 변수의 device, dtype을 기준으로 맞춤
            if isinstance(exists, (nn.Parameter, torch.Tensor)):
                tgt_device, tgt_dtype = exists.device, exists.dtype
            else:
                tgt_device, tgt_dtype = default_device, default_dtype

            # ckpt 텐서를 모델 변수의 device, dtype으로 변환
            src = ckpt_sd[name].to(device=tgt_device, dtype=tgt_dtype)

            if target.shape == src.shape:
                if isinstance(exists, (nn.Parameter, torch.Tensor)):
                    getattr(parent, attr).data.copy_(src); loaded.append(name)
                else:
                    # 기존 속성이 없거나 버퍼가 아니면 버퍼로 등록
                    parent.register_buffer(attr, src.clone(), persistent=True); loaded.append(name)
            else:
                mismatched.append((name, tuple(src.shape), tuple(target.shape)))
                if allow_partial and isinstance(exists, (nn.Parameter, torch.Tensor)):
                    n = min(target.numel(), src.numel())
                    getattr(parent, attr).data.view(-1)[:n].copy_(src.view(-1)[:n]); partial.append(name)
                else:
                    # shape 교체. dtype, device는 현재 변수 기준을 유지
                    src_t = src.detach()  # 이미 tgt_device, tgt_dtype로 변환됨
                    if name in param_names:
                        setattr(parent, attr, nn.Parameter(src_t.clone(), requires_grad=True))
                    elif name in buffer_names:
                        if hasattr(parent, attr):
                            delattr(parent, attr)
                        parent.register_buffer(attr, src_t.clone(), persistent=True)
                    else:
                        setattr(parent, attr, src_t.clone())
                    overwrited.append(name)

        for k in ckpt_sd.keys():
            if k not in model_sd: unexpected.append(k)

    if verbose:
        print(f"[force_load] loaded={len(loaded)} partial={len(partial)} missing={len(missing)} mismatched={len(mismatched)} unexpected={len(unexpected)} OVERWRITED={len(overwrited)}")
        if verbose >= 2 and (mismatched or missing or unexpected):
            if mismatched: print("[force_load] mismatched (name, ckpt, model):", mismatched[:20], "..." if len(mismatched) > 20 else "")
            if missing:    print("[force_load] missing:", missing[:20], "..." if len(missing) > 20 else "")
            if unexpected: print("[force_load] unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")
    if strict and (missing or mismatched):
        raise RuntimeError(f"Strict load failed. missing={len(missing)}, mismatched={len(mismatched)}")
    return {"loaded": loaded, "partial": partial, "missing": missing, "mismatched": mismatched, "unexpected": unexpected, "overwrited": overwrited}



def build_lr_scheduler(optimizer, sched_cfg, total_steps_fallback):
    if optimizer is None or sched_cfg is None:
        return None
    stype = str(sched_cfg.get('type', 'cosine')).lower()
    warmup = int(sched_cfg.get('warmup', 0))
    min_lr_mult = float(sched_cfg.get('min_lr_mult', 0.01))
    total_steps = int(sched_cfg.get('total_steps', total_steps_fallback))

    if stype == 'cosine':
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            if step < warmup and warmup > 0:
                return max(1e-8, step / float(max(1, warmup)))
            # cosine decay from 1 → min_lr_mult over [warmup, total_steps]
            t = min(max(0, step - warmup), max(1, total_steps - warmup))
            cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, total_steps - warmup)))
            return min_lr_mult + (1.0 - min_lr_mult) * cos
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif stype == 'multistep':
        # default milestones at 60/80/90% of total steps, with warmup grafted via LambdaLR
        milestones = sched_cfg.get('milestones', [int(0.6*total_steps), int(0.8*total_steps), int(0.9*total_steps)])
        gamma = float(sched_cfg.get('gamma', 0.1))
        base = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if warmup <= 0:
            return base
        # wrap warmup using LambdaLR chained before MultiStep
        def warm_lambda(step):
            return max(1e-8, step / float(max(1, warmup))) if step < warmup else 1.0
        warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_lambda)
        # Create a SequentialLR to apply warmup then multistep
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm, base], milestones=[warmup])
    else:
        return None