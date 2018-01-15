class hyperparameterModel(object):
    def __init__(self):
        self.CNN1Size = 5
        self.CNN1FeatureSize = 32
        self.CNN2Size = 3  # use 0 for disable.
        self.CNN2FeatureSize = 32
        self.CNN3Size = 3  # use 0 for disable.
        self.CNN3FeatureSize = 64
        self.CNN4Size = 3  # use 0 for disable.
        self.CNN4FeatureSize = 128
        self.FullyConn1Size = 128  # use 0 for disable.
        self.FullyConn2Size = 64  # use 0 for disable.

    def paramstxt(self):
        outputname = str(self.CNN1Size) + 'x' + str(self.CNN1FeatureSize) + '-' + str(self.CNN2Size) + 'x' + str(
            self.CNN2FeatureSize) + '-' + str(self.CNN3Size) + 'x' + str(self.CNN3FeatureSize) + '-' + str(self.CNN4Size) + 'x' + str(
            self.CNN4FeatureSize) + '-F' + str(self.FullyConn1Size) + '-F' + str(self.FullyConn2Size)
        return outputname
