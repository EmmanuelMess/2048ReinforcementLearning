class LUT:
    def __init__(self, m, n, loc):
        super(LUT, self).__init__()
        self.table = [[0 for j in range(n[i])] for i in range(m)]
        self.loc = loc
        self.c = 15

    def calculate(self, s):
        r = 0
        for i in range(len(self.table)):
            index = 0
            for j in range(len(self.loc[i])):
                index += self.loc[i][j] * (self.c ** (j - 1))

            r += self.table[i][index]

        return r