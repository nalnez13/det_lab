import re
import yaml


__loader = yaml.SafeLoader
__loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def load_yaml_file(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=__loader)


def get_train_configs(file=None):
    default_configs = load_yaml_file('configs/default_settings.yaml')

    if file:
        custom_configs = load_yaml_file(file)
    if custom_configs:
        default_configs.update(custom_configs)

    return default_configs
