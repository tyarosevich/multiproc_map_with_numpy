# The general approach to mapping numpy arrays was taken from:
# https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

from multiprocessing import shared_memory
import numpy as np
import multiprocessing as mp
import os
from functools import partial as func_partial


def run_shared_mem_multiproc_map(input_data: tuple, input_fun, shared_mem_data: dict, settings: dict, kwargs=None):
    """
    A wrapper to use an arbitrary function with a multiprocessing mapping pool that arbitrary numpy arrays across all
    processes without copying using shared memory.
    CRITICAL: input_data is a tuple of iterables, even if it's a tuple of one list. If multiple map lists are necessary,
    it needs to be an iterable like a tuple to zip the unpacked tuple as an argument to starmap.
    CRITICAL: This pattern has done weird things with larger numpy arrays for me, causing unexplained system errors, but
    NOT when run from command line. Bizarre.
    CRITICAL: This will not work in windows.
    Parameters
    ----------
    input_data: tuple
        A tuple of iterables of equal length to pass to the mapping function.
    input_fun: function
        Some mapping function that satisfies all the requirements of knowing positional arguments, settings, numpy
        array keys etc.
    shared_mem_data: dict
        A dictionary of numpy arrays. The keys will be used as unique names for shared memory.
    settings: dict
        Meta settings for the mutiprocessing job, e.g. cores to keep free.
    kwargs: dict
        Any additional arguments for input_fun. Note that these arguments WILL be copied into each shared process.

    Returns
    -------
    list
        A list of the mapped output. Can be just about anything.

    """
    if os.name != 'posix':
        raise NotImplementedError(
            'This function relies on Linux forks() for multiprocessing. Not intended for use on Windows.')

    # Ensure all input iterables are the same length.
    input_lengths = []
    for input_iter in input_data:
        input_lengths.append(len(input_iter))
    assert len(set(input_lengths)) == 1

    # Create the shared memory arrays from the data. dct_shm_params stores the shared mem parameters that need to be
    # passed to the function to be accessed during multiprocessing. dct_shm_address stores the returned shared mem
    # object in order to close and unlink all memory after processing.
    dct_shm_params = dict.fromkeys(shared_mem_data.keys(), {'shape': None, 'dtype': None})
    dct_shm_address = dict.fromkeys(shared_mem_data.keys())
    for shm_name, np_array in shared_mem_data.items():
        dct_shm_params[shm_name]['shape'] = np_array.shape
        dct_shm_params[shm_name]['dtype'] = np_array.dtype
        dct_shm_address[shm_name] = _create_shared_memory_nparray(np_array, shm_name)

    # A runner function is created in order to properly map shared memory inputs, multiple positional inputs, and
    # additional arguments to the actual mapping function.
    runner = func_partial(_map_fun_wrapper, input_fun=input_fun, shm_params=dct_shm_params, kwargs=kwargs)

    num_cores = mp.cpu_count() - settings['num_cores_kept_free']
    if num_cores < 1:
        raise ValueError("The number of cores left free results in 0 cores for multiprocessing.")

    try:
        num_input_lists = len(input_data)
        with mp.Pool(processes=num_cores) as pool:
            if num_input_lists == 1:
                multi_proc_output = list(pool.map(runner, input_data[0]))
            elif num_input_lists > 1:
                multi_proc_output = list(pool.starmap(runner, list(zip(*input_data))))
    except Exception as e:
        raise e
    finally:
        try:
            for shm_obj in dct_shm_address.values():
                shm_obj.close()
                shm_obj.unlink()
        except Exception as e:
            print("Somehow, the shared memory failed to close. Danger, Will Robinson. "
                  "Recommend terminating interpreter/IDE session and closing anything even remotely related to this "
                  "session.")
            raise e

    return multi_proc_output

def _map_fun_wrapper(*input_data: tuple, input_fun=None, shm_params=None, kwargs=None):
    """
    Wrapper function for the actual input mapping function. The *input_data positional parameters correspond to the map
    function arguments and are just passed on as a tuple to input_fun (input_fun must know what to do with this). NOTE,
    the named parameters input_fun and shm_params are NOT optional - they are named in order to accomodate unpacking
    input_data AND the map functions.
    Parameters
    ----------
    input_data: tuple
        The incoming map data. Can be a tuple of elements (map) or tuple of tuples (starmap), in which each element
        of the nested tuple is a positional argument for input_fun.
    input_fun: function
        The actual mapping function. Must know how to handle the input data and share memory data.
    shm_params: dict
        The shared memory data. Keys are the shm names, values are the numpy arrays.
    kwargs: dict
        Any additional settings for input_fun, which will be unpacked as keyword arguments.

    Returns
    -------
    The result of the mapping function. Can be just about anything.
    """

    # This initializes the shared memory objects on the other side of the pool.map/starmap function, and passes them
    # to the mapping function. It is assumed that this data corresponds correctly to the input function, i.e. the
    # input_fun will know the names of the shared memory numpy arrays (shm_name) and know what to do with them. Input
    # fun also needs to know how many positional arguments there are in order to unpack the input_data tuple.
    dct_shm_data_mp_side = dict.fromkeys(shm_params.keys(), {'mem_obj': None, 'ndarray': None})
    for shm_name, shm_data in shm_params.items():
        dct_shm_data_mp_side[shm_name]['mem_obj'] = shared_memory.SharedMemory(name=shm_name)
        dct_shm_data_mp_side[shm_name]['ndarray'] = np.ndarray(shape=shm_data['shape'], dtype=shm_data['dtype'],
                                                               buffer=dct_shm_data_mp_side[shm_name]['mem_obj'].buf)

    if kwargs:
        output = input_fun(*input_data, dct_shm_data_mp_side, **kwargs)
    else:
        output = input_fun(*input_data, dct_shm_data_mp_side)

    return output


def _create_shared_memory_nparray(input_data: np.ndarray, shared_name: str):
    """
    Write a numpy array to shared memory.
    Parameters
    ----------
    input_data: numpy.ndarray
        The array to be written to shared memory.
    shared_name: str
        The unique name of the shared memory address object.

    Returns
    -------
    The shared memory object, which can be used to unlink memory.
    """

    d_size = np.dtype(input_data.dtype).itemsize * np.prod(input_data.shape)
    try:
        shm = shared_memory.SharedMemory(create=True, size=d_size, name=shared_name)
    except FileExistsError:
        Warning ("Shared memory already exists. Attempting to release it and continue.")
        _release_shared(shared_name)
        shm = shared_memory.SharedMemory(create=True, size=d_size, name=shared_name)

    # numpy array on shared memory buffer
    dst = np.ndarray(shape=input_data.shape, dtype=input_data.dtype, buffer=shm.buf)
    dst[:] = input_data[:]

    return shm


def _release_shared(name):
    """
    Release an existing shared memory numpy array.
    Parameters
    ----------
    name: str
        The unique name of the shared memory address object.

    Returns
    -------
    None
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()  # Free and release the shared memory block
        print('Shared memory "{}" is hypothetically released'.format(name))
    except Exception as e:
        print("Failed to release the shared memory.")
        raise e
