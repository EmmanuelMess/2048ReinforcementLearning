class LUT:
    def __init__(self, m, n, loc, c, alpha):
        super(LUT, self).__init__()
        self.table = [[0 for j in range(n[i] * (c ** n[i]))] for i in range(m)]
        self.loc = loc
        self.c = c
        self.alpha = alpha

    def index(self, s, i):
        index = 0
        for j in range(len(self.loc[i])):
            index += int(s[self.loc[i][j]] * (self.c ** (j - 1)))
        return index

    def calculate(self, s):
        r = 0
        for i in range(len(self.table)):
            r += self.table[i][self.index(s, i)]

        return r

    def update(self, s, s_prime, reward):
        fs = self.calculate(s)
        fs_prime = self.calculate(s_prime)
        update = self.alpha * (reward + fs_prime - fs)

        for i in range(len(self.table)):
            self.table[i][self.index(s, i)] += update
