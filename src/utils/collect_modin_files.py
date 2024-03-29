import pandas as pd 
import subprocess
import glob 
import os
import sys 
import csv    
import yaml 
import time 

if __name__ == "__main__":
    start = time.time()
    wf_config_file = sys.argv[1]

    with open(os.path.join('/workspace/configs',os.path.basename(wf_config_file)),'r') as file:
        config = yaml.safe_load(file)

    worker_ips = config['env']['node_ips'][1:]
    data_path = config['data_preprocess']['output_data_path']
    output_data_path = '/workspace/data/' + data_path
    file_name = 'processed_data.csv'

    for i, ip in enumerate(worker_ips):
        new_file_name = file_name.split('.')[0] +'_'+ str(i+1) + '.' + file_name.split('.')[1]
        command = f"scp -o StrictHostKeyChecking=no {ip}:{output_data_path}/{file_name} {output_data_path}/{new_file_name}"
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    time.sleep(5)
    files = sorted(glob.glob(f'{output_data_path}/*.csv'))
    print(files)

    file_with_names = ''

    for file in files: 
        with open(file, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
        if 'year' in fieldnames:
            col_names = fieldnames
            file_with_names = file 
    
    print(col_names)
    
    df = []
    for file in files:
        if file == file_with_names:
            csv = pd.read_csv(file)
        else:
            csv = pd.read_csv(file, header=None, names=col_names)
        df.append(csv)

    data = pd.concat(df)
    print(data.shape)

    for f in files:
        os.remove(f)

    data.to_csv(f"{output_data_path}/{file_name}", index=False)
    print("this script took %.1f seconds" % ((time.time()-start)))

