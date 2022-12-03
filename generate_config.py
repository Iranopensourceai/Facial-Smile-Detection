from configparser import ConfigParser

config = ConfigParser()

config["DEFAULT"] = {
    "image_width" : 32,
    "image_height" : 32,
    "image_depth" : 1,
    "num_of_classes" : 2
}


with open("config_file.ini", "w") as f:
  config.write(f)
