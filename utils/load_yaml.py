import yaml


def load_yaml_file(file):
    with open(file, 'r') as f:
        return yaml.load(f, yaml.FullLoader)
