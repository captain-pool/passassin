import json
import os
initiated = False
class settings:
    def init(filename = None):
        global settings,initiated
        if filename:
            with open(os.path.abspath(filename)) as f:
                settings = json.load(f)
        else:
            settings  = {}
        initiated = True
    def get(key,default=None):
        global settings,initiated
        assert initiated
        return settings.get(key,default)
    def set(key,value):
        global settings,initiated
        assert initiated
        settings[key] = value

