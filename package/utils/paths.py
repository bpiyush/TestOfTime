"""Path utils."""


def get_data_root_from_hostname():
    import socket

    data_root_lib = {
        "diva": "/ssd/pbagad/datasets/",
        "node": "/var/scratch/pbagad/datasets/",
        "fs4": "/var/scratch/pbagad/datasets/",
    }
    hostname = socket.gethostname()
    hostname = hostname[:4]
    
    data_root = data_root_lib[hostname]
    return data_root
    
