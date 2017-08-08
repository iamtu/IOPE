import numpy as np
def np_array_to_string(a):
    _str = ''
    for i in xrange(len(a)-1):
        _str += str(a[i]) + ',';
    _str += str(a[-1]) + '\n'
    return _str

a = np.array([1,2,3]);
b = np.array([4,5,6]);

f = open('test.txt','w');
f.write(np_array_to_string(a));
f.write(np_array_to_string(b));
f.close();

