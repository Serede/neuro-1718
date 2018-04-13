import numpy as np

class Alphabet:

    filename=None
    _data = None
    _size = None
    _resolution = None

    def __init__(self,filename):

        self.filename = filename

        with open(filename) as file:
            lines = file.read().splitlines()

        data = []

        try:
            for l in lines:
                if l.startswith("//"):
                    data.append([])
                else:
                    data[-1].extend(l.split())
        except:
            raise ValueError('Bad input file :( {}'.format(filename))

        self._data = np.asarray(data).astype(int)
        self._size , self._resolution = self._data.shape

    def export(self, n, errors,filename):
        data = np.tile(self._data, (n, 1))

        with open(filename,'w') as file:
            for i in range(data.shape[0]):
                r = np.random.choice(range(self._resolution),size=errors)
                data[i][r] = 1 - data[i][r]

                file.write(' '.join(data[i].astype(str)) + '\n')


        return data