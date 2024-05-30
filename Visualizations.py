import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from statistics import mean, stdev

def vis_cv():
    with open("test.txt") as f:
        contents = f.readlines()
        dict = {}
        for content in contents:
            temp = content.split(':')
            key = temp[0].strip("'")
            value = round(float(temp[1].strip(', \n')),6)
            dict[key] = value 
    title = "p values"      
    width = 18
    height = 6
    sns.set(rc = {'figure.figsize':(width,height)})
    sns.barplot(data=dict, color="#099c94").set(title=title)
    sns.set_style("dark")
    plt.show()
vis_cv()
    