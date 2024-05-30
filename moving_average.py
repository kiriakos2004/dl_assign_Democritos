import pandas as pd

file = "your name of file.csv"
width  = 10


x = file.split("clean.")
new_file = x[0]+"clean_mov_avg."+str(width)+"."+x[1]

def find_moving_average(file, width):
    global means
    series = pd.read_csv(file, header=0)
    series = series.drop(['Time [TIMESTAMP]', 'Vessel Hull Latitude Angle (Instrument GPS 1) - Thisseas [VALUE]','Vessel Hull Longitude Angle (Instrument GPS 1) - Thisseas [VALUE]'], axis=1)
    hello = series.rolling(window=width)
    means = hello.mean(numeric_only=False)
    means = means[width-1:]
    return(means)

def create_csv():
    means.to_csv(new_file)


if __name__ == "__main__":
    find_moving_average(file, width)
    create_csv()

