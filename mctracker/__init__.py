
def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()

def find_local_address():
    import netifaces
    addr_list = []
    for idx, iface in enumerate(netifaces.interfaces()):
        try:
            addr = netifaces.ifaddresses(iface)
            for a in addr[netifaces.AF_INET]:
                addr_list.append(a['addr'])
        except Exception:
            pass
    return addr_list

