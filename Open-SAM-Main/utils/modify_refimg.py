import os

out_root = 'outputs/fss-te'

datasets = ['fold0', 'fold1', 'fold2', 'fold3']
for dataset in datasets:
    datasetdir = os.path.join(out_root,dataset)
    
    for method in os.listdir(datasetdir):
        if 'refimg' not in method:
            continue
        
        refimg = method.split('_')[1]
        
        methoddir = os.path.join(datasetdir,method)
        log_path = os.path.join(methoddir, 'log.txt')
        
        print(methoddir)
        filedata = ""
        with open(log_path, 'r') as f:
            for line in f:
                if "All miou" in line:
                    filedata += line
                else:
                    x, y, z = tuple(line.split(" "))
                    if '/' not in y:
                        y = refimg + '/' + y
                    filedata += x + ' ' + y + ' ' + z
        print(filedata)
        
        # raise NameError
        with open(log_path ,'w') as f:
            f.write(filedata) 
        
    