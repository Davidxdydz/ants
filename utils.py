import time
import functools

enabled = True

def set_enabled(printing = True):
    global enabled
    enabled = printing

def perf(name = None, include_average = True):
    global enabled
    if not enabled:
        return lambda f : f
    def decorator(f):
        nonlocal name
        if name is None:
            name = f.__name__
        total = 0
        n = 0
        @functools.wraps(f)
        def wrapper(*args,**kwargs):
            nonlocal total
            nonlocal n
            start = time.time()
            result = f(*args,**kwargs)
            ms = (time.time()-start)*1000
            total += ms
            n+= 1
            output = f"{name}: {ms:.2f}ms"
            if include_average:
                output += f"\tavg: {total/n}ms"
            print(output)
            return result
        return wrapper
    return decorator