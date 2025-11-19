import aiohttp
import torch

from slime.utils.types import Sample
import wandb
import importlib

def load_function_from_path(function_path: str):
    module_path, func_name = function_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func

async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:] for t_log_prob, response_length in zip(teacher_log_probs, response_lengths)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs):
        sample.teacher_log_probs = t_log_probs
    
    acc_reward_path = args.custom_acc_reward
    if acc_reward_path is not None:
        acc_reward_func = load_function_from_path(acc_reward_path)
        acc_rewards = [acc_reward_func(sample.response, sample.label) for sample in samples]
        acc_avg_reward = sum(acc_rewards) / len(acc_rewards)
        wandb.log({"rollout/acc_avg_reward": acc_avg_reward})

    return teacher_log_probs, teacher_log_probs
