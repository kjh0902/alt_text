"""CUDA 전용 Gemma 4 E4B 로딩·한국어 라벨 점수·LoRA 초기화."""
from __future__ import annotations
import gc, hashlib, inspect, json, re, time, threading, sys, platform
from pathlib import Path
import torch
from filelock import FileLock, Timeout
from tqdm.auto import tqdm
from .utils import LABELS, environment, package_versions, seed_all, file_sha, digest, read_json, save_json, code_hash
from .images import IMAGE_POLICY
from .scoring import append_target, collate_inputs, target_logprobs

_ACTIVE_ENGINE=False


def require_cuda() -> dict:
    env=environment()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU 없음. CPU로 전환하지 않습니다. NVIDIA 데스크톱의 CUDA용 PyTorch 커널을 선택하세요.')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 미지원: 공통 dtype 변경 없이 중단합니다.')
    return env


def check_gpu_kernels() -> dict:
    env=require_cuda()
    import bitsandbytes as bnb
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    layer=bnb.nn.Linear4bit(64,32,bias=False,compute_dtype=torch.bfloat16,
                           quant_type='nf4',compress_statistics=True).to('cuda')
    for p in layer.parameters():p.requires_grad_(False)
    # 양자화 weight를 통과하는 입력 gradient 경로 검사
    x=torch.randn(2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):y=layer(x);loss=y.float().square().mean()
    loss.backward();torch.cuda.synchronize()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):raise RuntimeError('bitsandbytes backward 검사 실패')
    out={'environment':env,'nf4_forward_backward':True,'seconds':time.perf_counter()-started,
         'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
         'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    del layer,x,y,loss;gc.collect();torch.cuda.empty_cache()
    return out


def resolve_revision(cfg: dict) -> str:
    from huggingface_hub import HfApi
    p=Path(cfg['_work'])/'model_revision_lock.json';spec=cfg['model']
    if p.exists():
        old=read_json(p)
        if old['model_id']!=spec['id'] or (spec['revision']!='main' and spec['revision']!=old['requested_revision'] and spec['revision']!=old['revision']):
            raise RuntimeError('모델 revision 변경: 새 work_dir 필요')
        return old['revision']
    sha=HfApi().model_info(spec['id'],revision=spec['revision']).sha
    if not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{40}',sha):raise RuntimeError('실제 모델 commit SHA 확인 실패')
    save_json(p,{'model_id':spec['id'],'requested_revision':spec['revision'],'revision':sha})
    return sha


def tensor_state_hash(state: dict) -> str:
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        t=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class GemmaEngine:
    def __init__(self,cfg: dict):
        global _ACTIVE_ENGINE
        require_cuda()
        if _ACTIVE_ENGINE:raise RuntimeError('다른 GemmaEngine이 살아 있습니다. close() 또는 커널 종료 후 순차 실행하세요.')
        lockpath=Path.home()/'.cache'/'koaltq'/'cuda0.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
        self.lease=FileLock(str(lockpath));self.closed=False;self.model=None;self.processor=None;self.cfg=cfg
        try:self.lease.acquire(timeout=0)
        except Timeout:raise RuntimeError('다른 KoAltQ 프로세스가 GPU 사용 중입니다. 세 노트북을 순차 실행하세요.')
        _ACTIVE_ENGINE=True
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration
            signature=inspect.signature(Gemma4ForConditionalGeneration.forward)
            required={'logits_to_keep','mm_token_type_ids','image_position_ids','pixel_values'}
            if not required<=set(signature.parameters):raise RuntimeError('설치된 Gemma4 forward API가 검증된 API와 다릅니다.')
            self.revision=resolve_revision(cfg);seed_all(cfg['seed']);self.encode_lock=threading.RLock()
            q=cfg['model']
            if q['quantization']!='nf4' or q['double_quant'] is not True:raise ValueError('공통 양자화 설정 변경 미지원')
            quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
            cache=str(Path(cfg.get('model_cache_dir') or (Path(cfg['_work'])/'model_cache')))
            self.processor=AutoProcessor.from_pretrained(q['id'],revision=self.revision,cache_dir=cache)
            self.processor.tokenizer.padding_side='left'
            if self.processor.tokenizer.pad_token_id is None:raise RuntimeError('Gemma tokenizer pad_token_id 없음')
            if 'enable_thinking' not in str(self.processor.chat_template):raise RuntimeError('enable_thinking 공식 템플릿 확인 실패')
            ip=self.processor.image_processor
            if not hasattr(ip,'max_soft_tokens'):raise RuntimeError('이미지 processor max_soft_tokens 미지원')
            self.model=Gemma4ForConditionalGeneration.from_pretrained(q['id'],revision=self.revision,
                cache_dir=cache,quantization_config=quant,dtype=torch.bfloat16,
                device_map={'':0},attn_implementation=q['attention'])
            loaded=getattr(self.model.config,'_commit_hash',None)
            if loaded and loaded!=self.revision:raise RuntimeError('실제로 로딩된 commit SHA 불일치')
            if any(p.device.type!='cuda' for p in self.model.parameters()):raise RuntimeError('CPU/disk offload 감지: CUDA 전용 조건 위반')
            for p in self.model.parameters():p.requires_grad_(False)
            self.model.eval();self.adapter_active=False
            self.label_tokens={lab:self.processor.tokenizer.encode(lab,add_special_tokens=False) for lab in LABELS}
            for lab,ids in self.label_tokens.items():
                if not ids or self.processor.tokenizer.decode(ids,skip_special_tokens=False)!=lab:
                    raise RuntimeError(f'라벨 전체 토큰 round-trip 실패: {lab}')
            self.identity={'model_id':q['id'],'revision':self.revision,'quantization':q,
                'python':sys.version,'platform':platform.platform(),'torch_cuda':torch.version.cuda,
                'gpu_name':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),
                'implementation_sha256':code_hash(Path(cfg['_package_root'])),
                'packages':package_versions(),'image_policy':IMAGE_POLICY,'input':cfg['input'],
                'processor_class':type(self.processor).__name__,'image_processor_class':type(ip).__name__,
                'image_processor_config':ip.to_dict(),'chat_template_sha256':digest(self.processor.chat_template),
                'label_tokens':self.label_tokens,'base_frozen_dtype_policy':'native_bf16_no_global_fp32_cast'}
            self.identity = json.loads(json.dumps(self.identity, ensure_ascii=False))
            save_json(Path(cfg['_work'])/'actual_model_api.json',{'identity':self.identity,
                'forward_signature':str(signature),'model_class':type(self.model).__name__,
                'parameter_devices':sorted(set(str(p.device) for p in self.model.parameters())),
                'model_memory_gib':self.model.get_memory_footprint()/2**30})
            print('실제 모델 revision:',self.revision);print('라벨 전체 토큰:',self.label_tokens)
            print('모델 메모리 GiB:',round(self.model.get_memory_footprint()/2**30,3))
        except BaseException:
            self.close();raise

    def encode(self,text: str,image) -> dict:
        # 공유 tokenizer 상태 보호; 이미지 디코딩은 CPU 스레드에서 병렬 실행
        with self.encode_lock:return self._encode_locked(text,image)

    def _encode_locked(self,text: str,image) -> dict:
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                                    enable_thinking=self.cfg['input']['enable_thinking'])
        # 실제 토크나이저 경계 검증: 첫 한국어 라벨 토큰까지 prompt와 병합되지 않는지 확인
        prefix=self.processor.tokenizer.encode(rendered,add_special_tokens=False)
        for lab,ids in self.label_tokens.items():
            full=self.processor.tokenizer.encode(rendered+lab,add_special_tokens=False)
            if full!=prefix+ids:raise RuntimeError(f'prompt/label 토큰 경계 변경: {lab}. 임의 위치 보정 없이 중단합니다.')
        batch=self.processor(text=rendered,images=image,return_tensors='pt',
            add_special_tokens=False,truncation=False,padding=False,
            max_soft_tokens=self.cfg['input']['max_soft_tokens'])
        base={k:v.cpu() for k,v in batch.items()}
        length=base['input_ids'].shape[1]
        if length+max(map(len,self.label_tokens.values()))>self.cfg['input']['max_input_tokens']:
            raise ValueError(f'입력 토큰 상한 초과({length}). 문맥/해상도 자동 축소 없음; 공통 조건 변경 시 세 실험 재실행 필요')
        if base['input_ids'].shape[0]!=1:raise RuntimeError('레코드당 입력 배치 크기 오류')
        return base

    def to_gpu(self,batch: dict) -> dict:
        return {k:v.to(device='cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,non_blocking=True)
                for k,v in batch.items()}

    def forward_targets(self,pairs: list[tuple[dict,str]]):
        items=[];targets=[]
        for base,lab in pairs:
            x,t=append_target(base,self.label_tokens[lab]);items.append(x);targets.append(t)
        batch,labels=collate_inputs(items,self.processor.tokenizer.pad_token_id,targets)
        k=int((labels!=-100).sum(-1).max())+1
        batch=self.to_gpu(batch);labels=labels.to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model(**batch,use_cache=False,logits_to_keep=k)
        return target_logprobs(output.logits,labels)

    def score_reference(self,bases: list[dict],progress: bool=False) -> list[dict]:
        # 기존 단건 n+1 로짓 방식과 독립적인 직접 gather 계산
        result=[];self.model.eval()
        with torch.no_grad(), tqdm(total=len(bases)*len(LABELS),desc='기준 라벨 점수',disable=not progress) as bar:
            for base in bases:
                scores={}
                for lab in LABELS:
                    ids=self.label_tokens[lab];x,_=append_target(base,ids);x=self.to_gpu(x)
                    with torch.autocast('cuda',dtype=torch.bfloat16):o=self.model(**x,use_cache=False,logits_to_keep=len(ids)+1)
                    logits=o.logits[:,:-1,:].float()
                    if logits.shape[1]!=len(ids):raise AssertionError('reference 전체 라벨 토큰 정렬 오류')
                    target=torch.tensor([ids],device='cuda');lp=torch.log_softmax(logits,-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
                    if not bool(torch.isfinite(lp).all()):raise FloatingPointError('reference 점수 오류')
                    scores[lab]={'sum_logprob':float(lp.sum()),'mean_logprob':float(lp.mean()),'n_tokens':len(ids)}
                    bar.update(1)
                result.append(scores)
        return result

    def score(self,bases: list[dict],candidate_batch: int,progress: bool=False) -> list[dict]:
        self.model.eval();out=[{} for _ in bases]
        tasks=[(i,lab,b) for i,b in enumerate(bases) for lab in LABELS]
        with torch.no_grad(), tqdm(total=len(tasks),desc='비교 라벨 점수',disable=not progress) as bar:
            for start in range(0,len(tasks),candidate_batch):
                part=tasks[start:start+candidate_batch]
                sums,means,ns=self.forward_targets([(b,lab) for _,lab,b in part])
                for j,(i,lab,_) in enumerate(part):
                    if lab in out[i]:raise AssertionError('후보 중복 처리')
                    out[i][lab]={'sum_logprob':float(sums[j]),'mean_logprob':float(means[j]),'n_tokens':int(ns[j])}
                bar.update(len(part))
        if any(set(s)!=set(LABELS) for s in out):raise AssertionError('후보/레코드 순서 연결 오류')
        return out

    def generate_layer1(self,bases: list[dict],max_new_tokens: int | None = None) -> list[str]:
        if self.adapter_active or hasattr(self.model,'peft_config') or any('lora_' in n for n,_ in self.model.named_parameters()):
            raise RuntimeError('LoRA 모델의 공통 Layer 1 생성 금지. 새 기본 모델을 로딩하세요.')
        self.model.eval();batch,_=collate_inputs(bases,self.processor.tokenizer.pad_token_id)
        n=batch['input_ids'].shape[1]
        limit=self.cfg['layer1']['max_new_tokens'] if max_new_tokens is None else int(max_new_tokens)
        if limit<1:raise ValueError('Layer1 max_new_tokens 오류')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.generate(**self.to_gpu(batch),max_new_tokens=limit,
                                       do_sample=False,use_cache=True)
        return self.processor.tokenizer.batch_decode(output[:,n:],skip_special_tokens=True)

    def attach_lora(self) -> dict:
        if self.adapter_active:raise RuntimeError('이미 LoRA가 활성화되었습니다. 기본 모델에서 별도로 시작해야 합니다.')
        from peft import LoraConfig,get_peft_model
        spec=self.cfg['lora'];suffixes=set(spec['target_suffixes'])
        names=[n for n,_ in self.model.named_modules() if re.fullmatch(r'model\.language_model\.layers\.\d+\.self_attn\.[a-z_]+',n)
               and n.rsplit('.',1)[-1] in suffixes]
        if not names or set(n.rsplit('.',1)[-1] for n in names)!=suffixes:
            raise RuntimeError(f'설치 모델의 지정 언어 self-attention 모듈 확인 실패: {names}')
        seed_all(self.cfg['seed'])
        lc=LoraConfig(r=spec['r'],lora_alpha=spec['alpha'],lora_dropout=spec['dropout'],
                      bias='none',target_modules=names,init_lora_weights=True)
        self.model=get_peft_model(self.model,lc);self.adapter_active=True
        # frozen BF16 유지: prepare_model_for_kbit_training의 전체 FP32 캐스팅 대신 비재진입 체크포인팅
        if self.cfg['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.config.use_cache=False
        trained={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        if not trained or any('lora_' not in n or '.language_model.' not in n for n in trained):
            raise AssertionError('LoRA 외 파라미터/비전 파라미터 학습 감지')
        initial={n:p.detach().cpu().clone() for n,p in trained.items()}
        initial_hash=tensor_state_hash(initial)
        path=Path(self.cfg['_work'])/'lora_initialization.json'
        info={'sha256':initial_hash,'target_modules':names,'trainable_parameters':sum(p.numel() for p in trained.values()),
              'trainable_names':list(trained),'lora_config':spec,'seed':self.cfg['seed']}
        if path.exists() and read_json(path)!=info:raise RuntimeError('실험 간 LoRA 초기화/적용 모듈 불일치')
        if not path.exists():
            save_json(path,info)
            from safetensors.torch import save_file
            save_file(initial,str(path.with_suffix('.safetensors')))
        self.initial_hash=initial_hash
        print('LoRA 적용 모듈:', len(names), '개 | 예:', ', '.join(names[:2]));print('학습 가능 파라미터:',info['trainable_parameters'])
        print('LoRA 초기화 SHA256:',initial_hash)
        return info

    def adapter_state(self) -> dict:
        from peft import get_peft_model_state_dict
        if not self.adapter_active:raise RuntimeError('LoRA 미활성')
        return {k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(self.model).items()}

    def load_adapter_state(self,state: dict) -> None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model,state)
        if tensor_state_hash(self.adapter_state())!=tensor_state_hash(state):raise RuntimeError('어댑터 복원 불일치')

    def fingerprint(self) -> str:
        return tensor_state_hash(self.adapter_state()) if self.adapter_active else digest(self.identity)

    def save_adapter(self,path: Path) -> None:
        if not self.adapter_active:raise RuntimeError('저장할 어댑터 없음')
        self.model.save_pretrained(str(path),safe_serialization=True)
        save_json(path/'koaltq_adapter.json',{'adapter_sha256':self.fingerprint(),'model_revision':self.revision,
                                          'initialization_sha256':self.initial_hash})

    def close(self) -> None:
        global _ACTIVE_ENGINE
        if getattr(self,'closed',False):return
        self.closed=True;self.model=None;self.processor=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if hasattr(self,'lease'):self.lease.release()
        _ACTIVE_ENGINE=False

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
