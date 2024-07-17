import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.properties')
    return config

config = load_config()

