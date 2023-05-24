import pandas as pd

class Environment:
    def __init__(self, normalizingType, fileName) -> None:
        self.lineNo = 0
        self.data = pd.read_csv("data/matrix1/{}/{}.csv".format(normalizingType, fileName))

    def getFileMaxLine(self):
        return self.data.shape[0]
    
    def getNextState(self, currentLine):
        return (self.data.iloc[currentLine + 1].values[:-1],  self.data.iloc[currentLine + 1].values[-1])

    def getCurrentState(self, currentLine):
        return (self.data.iloc[currentLine].values[:-1],  self.data.iloc[currentLine].values[-1])


if __name__ == "__main__":
    # environment = Environment("origin", "OK_http-params-normal")
    # print(environment.getNextState(1))
    data = pd.read_csv("data/matrix1/origin/OK_http-params-normal.csv")
    for idx, item in data:
        replay_buffer.add(data.iloc[idx].values[:-1], data.iloc[idx].values[1],0 if data.iloc[idx].values[1] == 0 else 10, data.iloc[idx + 1].values[:-1], True)
