This is a simple wrapper to generalize the use of shared memory to allow numpy arrays to be shared across multiple processes without copying. Thought it might be useful.

Some notes:
(1) This depends on the forks() functionality of Linux/Unix machines - it will not work on Windows. Full stop.
(2) Many input mapping iterables can be input and thus starmap will be used, but the mapping function must be aware of the number of positional arguments.
(3) Python >= 3.8
(4) I ran into strange problems with larger (>2GB) numpy arrays wherein it WAS working, but past a certain size, system errors would occur. This was not a problem of exceeding shared memory.
(5) Numpy arrays full of non-numerical objects didn't seem to work. I believe the underlying library requires contiguous memory.
(6) Working with shared memory in multiprocessing can always be a bit risky. Crashes have occurred, so please consider yourself warned. The author of this script is not responsible for any
    outcome. 

Example usage:
```
my_ndarray = np.random.random((100,))
shared_data = {'my_data': my_ndarray}
input_data = (np.random.random((10000,)), ) # Input must be a tuple of some iterable.
mp_settings = {'num_cores_kept_free': 2}

def mapping_fun(input, shared_data):
    my_uncopied_array = shared_data['my_data']['ndarray']
    return input + my_uncopied_array

start = time()
result = utils_multiprocess.run_shared_mem_multiproc_map(input_data, mapping_fun, shared_data, mp_settings)
end = time()
print('Time taken to run multiprocessing map: {} seconds'.format(end - start))
```
