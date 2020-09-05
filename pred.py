for i in range(0, 11):
    with open("{}.pbs".format(i), "w") as f:
        f.write('#PBS -N {}.pbs'.format(i) + '\n')
        f.write('#PBS -o /ghome/yujr/PBS/FISM_fastloss/{}.out'.format(i) + '\n')
        f.write('#PBS -e /ghome/yujr/PBS/FISM_fastloss/{}.err'.format(i) + '\n')
        f.write('#PBS -l nodes=1:gpus=1:S' + '\n')
        f.write('#PBS -r y' + '\n')
        f.write('#PBS  -m abef' + '\n')
        f.write('cd $PBS_O_WORKDIR'+ '\n')
        f.write('echo Time is \'data\''+ '\n')
        f.write('echo Directory is $PWD'+ '\n')
        f.write('echo This job runs on following nodes:'+ '\n')
        f.write('echo -n \"Node:\"'+ '\n')
        f.write('cat $PBS_NODEFILEs'+ '\n')
        f.write('echo -n \"Gpus:\"'+ '\n')
        f.write('cat $PBS_GPUFILE'+ '\n')
        f.write('startdocker -P /ghome/yujr/ -D /gdata/yujr/ -c \"python -u /ghome/yujr/NeuRec/main.py --lambda=0 --gamma=0 --alpha={} --init_method=tnormal\"  bit:5000/deepo_9'.format(0.1*i)+'\n')

