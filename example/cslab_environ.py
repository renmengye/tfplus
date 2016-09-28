import sys
import os
import socket

sys.path.insert(0, '../tfplus')

if os.path.exists('/u/mren'):
    hostname = socket.gethostname()
    if hostname.startswith('guppy'):
        sys.path.insert(
            # 0, '/pkgs/tensorflow-gpu-6July2016')
            0, '/pkgs/tensorflow-gpu-0.9.0')
            # 0, '/pkgs/tensorflow-gpu-0.8.0')
        pass
    else:
        sys.path.insert(
            0, '/pkgs/tensorflow-cpu-0.8.0')
        pass
    pass
