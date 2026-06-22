import torch


def unwrap_model(model):
    # 去检查传入的 model 有没有一个叫 "module" 的属性（hasattr）
    # 如果有（说明这个模型当前正处于多卡 DDP 包装状态），它就返回里面的真实核心网络 model.module。
    # 如果没有（说明是普通的单卡模型），它就原封不动地返回本身。
    # 不管模型外面套了多少层多卡的塑料壳，这个函数都能一键直达，拿到最纯粹的模型本体。
    return model.module if hasattr(model, "module") else model


def model_state_dict(model):
    # 剥离多卡属性，只抽取权重字典
    return unwrap_model(model).state_dict()


def load_model_state(model, state_dict):
    target_model = unwrap_model(model)  # 先把模型的外壳去掉
    clean_state_dict = {}
    for key, value in state_dict.items():
        # 如果权重键名是以 "module." 开头的，就把这个前缀切片裁掉
        clean_key = key[len("module."):] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value
    # 用干净的、无前缀的权重喂给单卡本体
    target_model.load_state_dict(clean_state_dict)


def load_checkpoint(path, map_location):
    # 通过传递这个参数（例如 map_location='cpu'），可以强行把权重拉回到当前指定的硬件内存中。
    return torch.load(path, map_location=map_location)
