# from settings import params
# from env.environment import Environment
# # from settings import args as parses
# import csv
# env=Environment(params)
# for i in range(1000):
#     env.step()
from settings import params
import csv
import os
dir_path='log/'+params["env_name"] +'/'+params["exp_name"]
if not os.path.exists(dir_path):
        os.makedirs(dir_path)