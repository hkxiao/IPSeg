import os

root = 'outputs/'

for type in os.listdir(root):
    kinddir = os.path.join(root,type)
    
    for dataset in os.listdir(kinddir):
        datasetdir = os.path.join(kinddir, dataset)
        
        for method in os.listdir(datasetdir):
            methoddir = os.path.join(datasetdir, method)
            

            if methoddir.endswith('SD_0.1_[0.3, 0.2, 0.1]'):
                dirlist = method.split('erosion')
                
                newdir  = datasetdir +'/'+ dirlist[0]+'SD_0.1_[0.3, 0.2, 0.1]_erosion' + dirlist[1].replace('_SD_0.1_[0.3, 0.2, 0.1]','')
                # print(methoddir, newdir)
                # raise NameError
                os.rename(methoddir, newdir)
        