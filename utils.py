import time
import functools

enabled = True

def set_enabled(printing = True):
    global enabled
    enabled = printing

def perf(name = None, include_average = True,skip = 0):
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
            nonlocal skip
            start = time.time()
            result = f(*args,**kwargs)
            ms = (time.time()-start)*1000
            if skip <= 0:
                total += ms
                n+= 1
            else:
                skip -= 1
            output = f"{name}: {ms:.2f}ms"
            if include_average:
                if n == 0:
                    output += f"\tavg: ---ms"
                else:
                    output += f"\tavg: {total/n}ms"
            print(output)
            return result
        return wrapper
    return decorator