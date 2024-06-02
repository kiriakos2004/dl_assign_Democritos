import matplotlib.pyplot as plt
import seaborn as sns
import ast

def vis_cv():
    with open("P_VALUES_50_percent.txt") as f:
        contents = f.read()
        # Converting the string content to a dictionary
        data_dict = ast.literal_eval(contents)
    
    title = "kNN 50% p values" 
    width = 18
    height = 8
    
    categories = list(data_dict.keys())
    values = list(data_dict.values())
    
    sns.set(rc={'figure.figsize':(width, height)})
    sns.set_style("dark")
    
    plt.figure(figsize=(width, height))
    ax = sns.barplot(x=categories, y=values, color="#099c94")
    ax.set_title(title)
    plt.xticks(rotation=90)
    
    plt.tick_params(axis='both', which='major', labelsize=6)
    
    current_ticks = ax.get_xticks()
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    ax.set_xticks(current_ticks)
    ax.set_xticklabels(current_labels)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    
    plt.show()

vis_cv()

    