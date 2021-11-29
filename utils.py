import configparser
from pathlib import Path
import os.path
import time

#Args: Name of csv file, array to append to file
#Appends array to csv
def array_to_csv(name, row):
    file = open(str(name) + '.csv', 'a')
    row = ", ".join(map(str, row)) + '\n'
    file.write(row)
    file.close()

def straight_to_csv(file_name, row):
    t = [time.time()]
    vals = list(row.values())
    flt_vals = []
    for item in vals:
        flt_vals.append(float(item))
  
    flt_vals = t + flt_vals
    print(flt_vals)
    array_to_csv(file_name, flt_vals)
    
    

def make_csv(file_name, row):
    title = ['time']
    keys = list(row.keys())
    title = title + keys
    print(title)
    array_to_csv(file_name, title)
    
    


def setup_json(file_name):
    print("To be implemented")

def dict_to_json(file_name, data):
    print("To be implemented")

#Args: name of configuration file
#Returns: dictionary of configuration data
def load_config(file_name):
    cp = configparser.ConfigParser()
    cp.read(file_name)
    dictionary = {}
    for section in cp.sections():
        dictionary[section] = {}
        for option in cp.options(section):
            dictionary[section][option] = float(cp.get(section, option))
    del cp
    return dictionary

#Args: Value to map, minimum input, maximum input, minimum output, maximum output,
def from_to(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#Args: csv filename
#Returns: new filename if original filename already exists
def next_log(file_name, extension):
    file_num = 1
    while os.path.isfile(file_name + extension):
        file_root = file_name.rsplit('_', 1)[0]
        file_name = file_root + '_' + str(file_num)
        file_num += 1
        
    print(file_name)
    return file_name

#print(107 * ((2 * (math.pi /60))* 0.73) * 2.23694)
