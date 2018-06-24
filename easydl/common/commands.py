def runTask():
    """
    this is a command. After install the package, the command will be available.

    this command is designed to run tasks automatically.

    suppose that one has 10 tasks to run but only has 8 GPUs, each task requires 1 GPU.  He has to
    run 8 tasks first, then checks whether tasks finish frequently, which is annoying.

    with this commoand ``runTask``, one can specify 10 commands to run in a file. every ``sleeptime`` seconds, it checks
    whether ``needGPU`` GPUs are available. if there are enough GPUs to run on, it get one line from ``file`` and executes
    the line. (if it succeeds in executing the line, that line will be removed.)

    the ``user`` argument is needed to query how many GPUs are in use (together with ``maxGPU``, it limits the number of
    GPUs one can use. This is often the case when GPUs are shared and one can't take up all GPUs)

    usage: runTask [-h] [--maxGPU MAXGPU] [--needGPU NEEDGPU] [--maxLoad MAXLOAD]
                [--maxMemory MAXMEMORY] [--sleeptime SLEEPTIME] [--user USER]
                file

    positional arguments:
      file                  file that contains one task per line

    optional arguments:
      -h, --help            show this help message and exit
      --maxGPU MAXGPU       maximum GPU to use by one user
      --needGPU NEEDGPU     number of GPUs per task/line
      --maxLoad MAXLOAD     GPU with load larger than this will be regarded as not
                            available
      --maxMemory MAXMEMORY
                            GPU with memory usage larger than this will be
                            regarded as not available
      --sleeptime SLEEPTIME
                            sleep time after executing one task/line
      --user USER           query how many GPUs user used so that it does not
                            violatethe limitation of maxGPU per user
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxGPU', type=int, default=1000, help='maximum GPU to use by one user')
    parser.add_argument('--needGPU', type=int, default=1, help='number of GPUs per task/line')
    parser.add_argument('--maxLoad', type=float, default=0.1, help='GPU with load larger than this will'
                                                                   ' be regarded as not available')
    parser.add_argument('--maxMemory', type=float, default=0.1,help='GPU with memory usage larger than this will'
                                                                   ' be regarded as not available')
    parser.add_argument('--sleeptime', type=float, default=60, help='sleep time after executing one task/line')
    parser.add_argument('--user', type=str, nargs=1,help='query how many GPUs user used so that it does not violate'
                                                 'the limitation of maxGPU per user')
    parser.add_argument('file', nargs=1, help='file that contains one task per line')
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
    user = args.user[0]
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