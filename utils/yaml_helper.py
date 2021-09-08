from torch.serialization import load
import yaml


def load_yaml_file(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def get_train_configs(file=None):
    default_configs = load_yaml_file('configs/default_settings.yaml')

    if file:
        custom_configs = load_yaml_file(file)
    if custom_configs:
        default_configs.update(custom_configs)

    return default_configs
