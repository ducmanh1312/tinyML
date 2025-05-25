import time
from functools import partial

# calc runtime method
def base_time_desc_decorator(method, desc='test_description'):
    def timed(*args, **kwargs):
        print(desc)
        
        start = time.time()
        
        # Run Method - use *args to properly pass self
        result = method(*args, **kwargs)
        
        print('Done! It took {:.2f} secs\n'.format(time.time() - start))
        
        if result is not None:
            return result
            
    return timed

def time_desc_decorator(desc): 
    return partial(base_time_desc_decorator, desc=desc)


@time_desc_decorator('this is description')
def time_test(arg, kwarg='this is kwarg'):
    time.sleep(3)
    print('Inside of time_test')
    print('printing arg: ', arg)
    print('printing kwarg: ',  kwarg)


@time_desc_decorator('this is second description')
def no_arg_method():
    print('this method has no argument')


if __name__ == '__main__':
    time_test('hello', kwarg=3)
    time_test(3)
    no_arg_method()
