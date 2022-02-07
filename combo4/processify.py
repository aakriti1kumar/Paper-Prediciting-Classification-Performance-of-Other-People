import multiprocessing as mp

def forked(fn):
    """
    Does not work on Windows (except WSL2), since the fork syscall is not supported here.
    fork creates a new process which inherits all of the memory without it being copied.
    Memory is copied on write instead, meaning it is very cheap to create a new process
    """
    def call(*args, **kwargs):
        ctx = mp.get_context('fork')
        q = ctx.Queue(1)
        is_error = ctx.Value('b', False)
        def target():
            try:
                q.put(fn(*args, **kwargs))
            except BaseException as e:
                is_error.value = True
                q.put(e)
        ctx.Process(target=target).start()
        result = q.get()
        if is_error.value:
            raise result
        return result
    return call

