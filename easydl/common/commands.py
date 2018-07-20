def runTask():
    """
    Usage:
        runTask [-h] <file> [--maxGPU=<int>] [--needGPU=<int>] [--maxLoad=<float>] [--maxMemory=<float>] [--sleeptime=<seconds>] --user=<string>

    Options:
      -h, --help     Show this screen.
      <file>            file that contains one task per line (should end with & to be run background)
      --maxGPU=<int>       maximum GPU to use by one user [default: 100].
      --needGPU=<int>     number of GPUs per task/line [default: 1].
      --maxLoad=<float>     GPU with load larger than this will be regarded as not available [default: 0.1].
      --maxMemory=<float>    GPU with memory usage larger than this will be regarded as not available [default: 0.1].
      --sleeptime=<seconds>    sleep time after executing one task/line [default: 180.0].
      --user=<string>           query how many GPUs user used so that it does not violatethe limitation of maxGPU per user


    this is a command. After install the package, the command will be available.

    this command is designed to run tasks automatically.

    suppose that one has 10 tasks to run but only has 8 GPUs, each task requires 1 GPU.  He has to
    run 8 tasks first, then checks whether tasks finish frequently, which is annoying.

    with this commoand ``runTask``, one can specify 10 commands to run in a file. every ``sleeptime`` seconds, it checks
    whether ``needGPU`` GPUs are available. if there are enough GPUs to run on, it get one line from ``file`` and executes
    the line. (if it succeeds in executing the line, that line will be removed.)

    the ``user`` argument is needed to query how many GPUs are in use (together with ``maxGPU``, it limits the number of
    GPUs one can use. This is often the case when GPUs are shared and one can't take up all GPUs)

    """
    from docopt import docopt
    args = docopt(runTask.__doc__)

    maxGPU = int(args['--maxGPU'])
    needGPU = int(args['--needGPU'])
    maxLoad = float(args['--maxLoad'])
    maxMemory = float(args['--maxMemory'])
    file = args['<file>']
    user = args['--user']
    sleeptime = float(args['--sleeptime'])

    from subprocess import Popen, PIPE
    import time
    from gpuutils import GPU, getGPUs, getAvailable, getAvailability, getFirstAvailable, showUtilization, __version__
    import random
    import os

    while True:
        with open(file) as f:
            lines = [line for line in f if line.strip()]
        if lines:
            while True:
                s = 'for x in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do ps -p $x -o pid,user | grep "%s"; done' % user
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