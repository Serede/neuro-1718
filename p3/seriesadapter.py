import numpy as np

class SeriesAdapter:

    filename=None
    _data = None
    _size = None
    _na = None
    _ns = None

    def __init__(self,filename,na,ns):

        self.filename = filename
        self._na = na
        self._ns = ns

        with open(filename) as file:
            lines = np.asarray(file.read().splitlines()).astype(float)

        self.size  = lines.size - na - ns + 1

        data = lines[:self.size]

        # Stack the rest of attributes
        for i in range(1,na+ns):
            data = np.vstack((data,lines[i:self.size +i]))

        self._data = data.T

    def export(self, filename):

        with open(filename,'w') as file:

            file.write('{} {}'.format(self._na, self._ns) + '\n')

            for v in self._data:
                file.write(' '.join(v.astype(str)) + '\n')