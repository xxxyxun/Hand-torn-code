###手写复现grpo算法，以qwen2.5-7B和GSM8K数据集为例
###参考trl库的实现
###要求数据格式与trl库一致
###包含'prompt'和'solution'两个字段{'prompt': ..., 'solution': ...}

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk, Dataset
from reward import reword_compute
from typing import Optional, Union, Callable, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

class MathDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        
        self.tokenizer = tokenizer
        self.data = load_dataset(path="json", data_files= data_path, split="train")
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample['answer']
        problem = sample['problem']
        return {'problem': problem, 'answer': answer}


@dataclass
class Samples:
    prompt_completion_ids: torch.Tensor
    completion_ids: torch.Tensor
    prompt_ids: torch.Tensor
    prompt: Any
    completion : Any
    solution: Any
    prompt_mask: Any #Optional[torch.LongTensor]
    completion_mask: Any #Optional[torch.LongTensor]
    action_mask: Any #Optional[torch.LongTensor]
    logits_to_keep: Union[int, torch.Tensor]
    completion_length: Any# int

class GRPOArgument:
    max_prompt_length: int = 256
    num_generations: int = 4
    max_completion_length: int = 256
    learning_rate: float
    beta: float = 0
    train_dataset: Dataset
    eval_dataset: Dataset
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    reward_weights: Optional[List[float]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int]= None
    gradient_accumulation_steps: int = 4
    epsilon_low: float = 0.1
    epsilon_high: float = 0.1
    epoch = 1
    num_iterations: int = 1
    batch_size: int = 2
    output_dir: str = './output'
    save_steps: int = 100

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class MYGRPOTrainer:
    def __init__(
            self, 
            reward_funcs,
            reward_tokenizers: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase],list ]]= None,
            model = None, 
            tokenizer: Optional[PreTrainedTokenizerBase]= None,
            args = GRPOArgument):
        
        self.args = args
        #加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        else:
            raise ValueError("Model should be a string path to the model.")
        
        assert isinstance(model, PreTrainedModel), "Model should be an instance of PreTrainedModel."
        self.model = model.to(self.args.device) 
        
        #加载参考模型
        self.ref_model = None
        if self.args.beta is not None:
            self.ref_model = deepcopy(self.model) #使用actor模型初始化
            self.ref_model.eval() #不参与梯度更新，只做推理

        #如果无tokenizer，则从模型配置中加载
        #保持左填充
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side="left")

        #加载奖励函数 转化成列表
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        #记录奖励函数名称，方便在训练时迭代取用
        self.reward_func_names = []
        for i, func in enumerate(reward_funcs):
            #如果是字符串，则代表奖励模型，加载模型
            if isinstance(func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(func, num_labels=1).to(self.args.device)
            #如果是函数，检查函数是否可调用
            elif not callable(func):
                raise ValueError(f"Reward function {func} is not callable.")
            
        self.reward_funcs = reward_funcs
        
        #如果有奖励权重，则转换成tensor
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError("Reward weights length must match number of reward functions.")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32, device=self.args.device)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32, device=self.args.device)

        #加载奖励模型toknenizer
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif not isinstance(reward_tokenizers, list):
            #保证reward_tokenizers数量与reward_funcs数量一致
            assert len(reward_tokenizers) == len(self.reward_funcs)
            reward_tokenizers = [reward_tokenizers]

        assert isinstance(reward_tokenizers, list)
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                #如果奖励函数是模型，且没说明tokenizer则自动加载对应的tokenizer
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path, padding_side="left")
                #检查tokenizer是否未设置填充标记(padding token) 
                #有些模型可能没有设置pad_token_id 则设置为eos_token
                #奖励模型不计算输入中填充令牌的奖励。
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token

                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer

        self.reward_tokenizers = reward_tokenizers
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  
        self.num_generations = args.num_generations 
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.train_dataset = args.train_dataset
        self.eval_dataset = args.eval_dataset
        self.epsilon_low = args.epsilon_low
        self.epsilon_high = args.epsilon_high

        #输入缓冲区，用于梯度累积
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        #当前步数
        self.update_steps = 0
    
    def generate_completions(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        solutions = [None] * len(prompts)

        if 'solution' in prompts:
            solutions = [sol for sol in inputs['solution']]

        max_length = self.args.max_completion_length + self.args.max_prompt_length

        SYSTEM_PROMPT = "You are a helpful assistant. Please answer the following questions."

        for prompt, solution in zip(prompts, solutions):

            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, 
                                                             {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            #复制prompt成一个组，得到多个生成
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, return_tensors='pt')
            prompt_ids = inputs['input_ids']
            #模型生成回答
            with torch.no_grad():
                prompt_completion_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_completion_length,
                                    temperature= self.args.temperature,
                                    top_p = self.args.top_p,
                                    top_k = self.args.top_k)
                
            #将生成的回答截断到最大长度 prompt_completion_ids形状[self.args.num_generations, len] ???
            if prompt_completion_ids.shape[1] > max_length:
                prompt_completion_ids = prompt_completion_ids[:, :max_length]
            #如果没到最大长度则用pad填充
            else:
                prompt_completion_ids = torch.cat([prompt_completion_ids, 
                                                torch.full((prompt_completion_ids.shape[0], max_length - prompt_completion_ids.shape[1]), 
                                                           self.tokenizer.pad_token_id, device=self.args.device)], dim=1)
            
            #提取填充后的prompt和响应部分
            completion_ids = prompt_completion_ids[:, prompt_ids.size(1):]
            prompt_ids = prompt_completion_ids[:, :prompt_ids.size(1)]

            #创建一个布尔张量（True/False），其中：True表示该位置的token不是填充token False表示该位置的token是填充token
            #再将布尔张量转换为整数张量True 1 False 0
            prompt_mask = (prompt_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            completion_mask = (completion_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)

            #创建一个动作掩码，标记哪些位置是有效的响应位置
            #标记需要计算奖励的位置 排除结束符(eos_token)和填充符(pad_token) 仅在有效生成token位置为1
            action_mask = (completion_ids.ne(self.tokenizer.eos_token_id) & completion_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)

            completion = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            samples = Samples(
                prompt_completion_ids=prompt_completion_ids,
                completion_ids = completion_ids,
                prompt_ids = prompt_ids,
                prompt = prompt,
                completion = completion,
                solution = solution,
                prompt_mask = prompt_mask,
                completion_mask = completion_mask,
                action_mask = action_mask,
                logits_to_keep = action_mask.size(1),
                completion_length = action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        #给 `logits_to_keep` 加 1，因为序列的最后一个 logits 后来会被排除。
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        # (B, seq_len-1, vocabulary_len), 排除最后一个对数几率：它对应于下一个标记预测。
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.args.temperature  # 除以温度参数
        # 计算每个标记的对数几率
        log_probs = F.log_softmax(logits, dim=-1) 
        # 提取实际生成token的概率 
        # input_ids[:, 1:] 是因为我们要预测下一个token，所以从第二个token开始
        #.unsqueeze(-1)：增加一个维度，形状变为[b, seq_len-1, 1]
        selected_logits = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        per_token_logps = selected_logits.squeeze(-1)
        return per_token_logps
    
    def generate_score_completions(self, inputs):

        self.model.eval()
        samples_list = self.generate_completions(inputs)

        batch_prompt_ids = []
        batch_prompt_mask = []
        batch_completion_ids  = []
        batch_completion_mask = []
        batch_advantages = []
        batch_old_per_token_logps = []
        batch_action_mask = []

        for samples in samples_list:
            prompt_completion_ids = samples.prompt_completion_ids # [num_generations, max_length]
            prompt_ids = samples.prompt_ids # [num_generations, max_prompt_length]
            completion_ids = samples.completion_ids  # [num_generations, max_completion_length]
            prompt_mask = samples.prompt_mask # [num_generations, max_prompt_length]
            completion_mask = samples.completion_mask # [num_generations, max_completion_length]
            action_mask = samples.action_mask # [num_generations, max_length]
            logits_to_keep = samples.logits_to_keep
            prompt = samples.prompt
            solution = samples.solution
            completion = samples.completion
            
            batch_prompt_ids.append(prompt_ids)
            batch_prompt_mask.append(prompt_mask)
            batch_completion_ids.append(completion_ids)
            batch_completion_mask.append(completion_mask)
            batch_action_mask.append(action_mask)

            #计算attention_mask
            attention_mask = prompt_mask + completion_mask

            with torch.no_grad():
                #计算每个token的对数概率
                old_per_token_logps = self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask, logits_to_keep)
                batch_old_per_token_logps.append(old_per_token_logps)

            #储存一组内的奖励
            rewards_per_func = torch.zeros(self.args.num_generations, len(self.reward_funcs), device=self.args.device)
            #将tokens转换为字符串
            completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            #将prompt复制num_generations次
            prompt_texts = [prompt] * self.args.num_generations
            #构建完整输出
            conversations = [prompt_text + completion_text for prompt_text, completion_text in zip(prompt_texts, completion_texts)]

            for i,(reward_func, reward_tokenizer) in enumerate(zip(self.reward_funcs, self.reward_tokenizers)):
                if isinstance(reward_func, PreTrainedModel):
                    with torch.inference_mode():
                        #如果是模型，则计算奖励
                        reward_inputs = reward_tokenizer(conversations, return_tensors='pt', padding=True).to(self.args.device)
                        rewards_per_func[:,i] = reward_func(**reward_inputs.to(self.args.device)).logits.squeeze(-1)
                else:
                    if isinstance(reward_func, Callable):
                        #如果是函数，则调用函数计算奖励
                        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
                        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                        solution = [solution] * self.args.num_generations
                        output_reward_func = reward_func(prompts=prompt_texts, completions=completion, completion_ids=completion_ids, **reward_kwargs)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[:,i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)

            if self.args.reward_weights is None:
                self.args.reward_weights = [1.0] * len(self.reward_funcs)
            if len(self.reward_weights) != len(self.reward_funcs):
                raise ValueError("Reward weights length must match number of reward functions.")
            
            rewards = rewards_per_func * torch.tensor(self.args.reward_weights, device=self.args.device).unsqueeze(0)

            mean_group_rewards = rewards.mean(dim=1)
            std_group_rewards = rewards.std(dim=1)
            #计算优势
            advantages = (mean_group_rewards - mean_group_rewards.mean()) / (std_group_rewards + 1e-8)
            batch_advantages.append(advantages)

        return {
            "prompt_ids": torch.stack(batch_prompt_ids),
            "prompt_mask": torch.stack(batch_prompt_mask),
            "completion_ids": torch.stack(batch_completion_ids),
            "completion_mask": torch.stack(batch_completion_mask),
            "advantages": torch.stack(batch_advantages),
            "old_per_token_logps": torch.stack(batch_old_per_token_logps),
            "action_mask": torch.stack(batch_action_mask),  
        }
    
    def compute_loss(self, model, inputs):
        """
        计算损失函数
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        action_mask = inputs["action_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        #只计算模型生成部分的logits
        logits_to_keep = completion_ids.size(1)
        #计算模型的logits
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep)
                    ref_per_token_logps = ref_per_token_logps * action_mask
            #使用k3估计的kl散度
                    per_token_kl = (
                        torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                    )

        advantages = inputs['advantages']

        #num_iterations == 1时old_per_token_logps == per_token_logps可以跳过计算
        old_per_token_logps = inputs['old_per_token_logps'] if self.num_generations > 1 else per_token_logps.detach() 
        
        #重要性采样 [batch_size * num_generations, num_actions]
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)

        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        
        #应用kl散度
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # shape: [batch_size * num_generations]
        loss = loss.mean()

        return loss
    
    def train_step(self, model, inputs, optimizer, step):
        """
        执行一次训练步骤
        """
        model.train()
        
        #计算损失
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps  # 梯度累积

        #反向传播
        loss.backward()
        
        if (step+ 1) % self.args.gradient_accumulation_steps == 0:

            #更新参数
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        
        #更新参数
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):

                inputs = self.generate_score_completions(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArgument()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    

  
    trainer = MYGRPOTrainer(model=model,
                          reward_funcs = [reword_compute],
                          args=args,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    
