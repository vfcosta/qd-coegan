from munch import Munch
import os

GAN_CONFIG_FILE = os.environ.get("GAN_CONFIG_FILE") or "default_config"
print(f"using the config file: {GAN_CONFIG_FILE}")
params = __import__(GAN_CONFIG_FILE).params
config = Munch.fromDict(params)
