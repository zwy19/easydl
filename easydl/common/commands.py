def runTask():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxGPU', type=int, default=1000)
    parser.add_argument('--needGPU', type=int, default=1)
    parser.add_argument('--maxLoad', type=float, default=0.1)
    parser.add_argument('--maxMemory', type=float, default=0.1)
    parser.add_argument('--sleeptime', type=float, default=60)
    parser.add_argument('--user', type=str)
    parser.add_argument('file', nargs=1)
    args = parser.parse_args()

    import cPickle
    from subprocess import Popen, PIPE

    import time

    from gpuutils import GPU, getGPUs, getAvailable, getAvailability, getFirstAvailable, showUtilization, __version__

    import random

    import os

    maxGPU = args.maxGPU
    needGPU = args.needGPU
    maxLoad = args.maxLoad
    maxMemory = args.maxMemory
    file = args.file[0]
    user = args.user
    sleeptime = args.sleeptime

    while True:
        with open(file) as f:
            lines = [line for line in f if line.strip()]
        if lines:
            while True:
                s = 'for x in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do ps -f -p $x | grep "%s"; done' % user
                p = Popen(s, stdout=PIPE, shell=True)
                ans = p.stdout.read()
                mygpu = len(ans.splitlines())
                deviceIDs = getAvailable(order='first', limit=needGPU, maxLoad=maxLoad, maxMemory=maxMemory,
                                         includeNan=False, excludeID=[], excludeUUID=[])
                find = False
                if mygpu < maxGPU and len(deviceIDs) >= needGPU:
                    os.system(lines[0].strip())
                    print('runing command(%s)' % lines[0].strip())
                    find = True
                time.sleep(sleeptime)
                if find:
                    break
            with open(file, 'w') as f:
                for line in lines[1:]:
                    f.write(line)
        else:
            break