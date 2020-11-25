class LUT:
    def __init__(self, m, n, loc, c, alpha):
        super(LUT, self).__init__()
        self.table = [[0 for j in range(n[i] * (c ** n[i]))] for i in range(m)]
        self.loc = loc
        self.c = c
        self.alpha = alpha

    def calculate(self, s):
        r = 0
        for i in range(len(self.table)):
            index = 0
            for j in range(len(self.loc[i])):
                index += int(s[self.loc[i][j]] * (self.c ** (j - 1)))

            r += self.table[i][index]

        return r

    def update(self, s, s_prime, reward):
        fs = self.calculate(s)
        fs_prime = self.calculate(s_prime)
        update = self.alpha * (reward + fs_prime - fs)

        for i in range(len(self.table)):
            index = 0
            for j in range(len(self.loc[i])):
                index += int(self.loc[i][j] * (self.c ** (j - 1)))

            self.table[i][index] += update
