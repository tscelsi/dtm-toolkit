import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
sns.set_style("whitegrid")
sns.set_context("paper")

colour_mapping = [
    "accounting",
    "agricultural activity",
    "asia and oceania",
    "america",
    "building and public works",
    "business operations and trade",
    "chemistry",
    "coal and mining industries",
    "criminal law",
    "cultivation of agricultural land",
    "defence",
    "demography and population",
    "deterioration of the environment",
    "distributive trades",
    "economic analysis",
    "economic conditions",
    "electrical and nuclear industries",
    "electronics and electrical engineering",
    "environmental policy",
    "europe",
    "executive power and public service",
    "food technology",
    "humanities",
    "iron, steel and other metal industries",
    "land transport",
    "marketing",
    "mechanical engineering",
    "natural and applied sciences",
    "natural environment",
    "oil and gas industry",
    "organisation of transport",
    "prices",
    "production",
    "regions and regional policy",
    "renewable energy",
    "research and intellectual property",
    "taxation",
    "technology and technical regulations",
    "wood industry",
    "world organisations"
]

def heatmap(X, save_path=None, **kwargs):
    """
    Plots simple heatmap by creating a matrix from the dataframe X
    """
    # set sequential colour palette to avoid confusion
    plt.clf()
    palette = sns.color_palette("viridis", as_cmap=True)
    g = sns.heatmap(data=X, center=0, cmap=palette, **kwargs)
    if save_path:
        print("saving heatmap...")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # else:
    #     print("showing heatmap...")
    #     plt.show()
    return plt, g

def single_topic_time_evolution(df, ax):
    ax.set_yticks([])
    x_domain = [x for x in range(1,len(df)+1)]
    x_labels = df.index.tolist()
    _max = max(df)
    assert len(x_domain) == len(x_labels)
#     plotter.set_xticks(x_domain)
#     plotter.set_xticklabels(x_labels)
#     max_val = 1 * df.max() + 5
    ax.fill_between(x_labels, df, -1*df, label="sup")

def time_evolution_plot(df, filename, title=None, scale=1, save=True, use_colour_mapping=False, fontsize='x-large'):
    """This function plots topic proportions over time. This function was adapted
    and is credited to the Muller-Hansen paper, the original code can be found in the
    following repository: https://github.com/mcc-apsis/coal-discourse.

    Args:
        df (pd.DataFrame): This is the dataframe that contains the data for plotting. The data should look
        similar to:
            topic_name          0         1        10  ...         7         8         9
            year                                       ...                              
            1997         1.388494  4.851258  0.682613  ...  2.151749  5.940288  5.549168
            1999         1.824987  4.308183  0.369598  ...  2.946666  4.397723  4.819061
            2000         2.007208  4.465947  0.954091  ...  7.107205  4.475813  5.143148
            2001         1.962062  5.138283  0.208519  ...  5.727354  4.850402  4.928038
            2002         2.296691  4.087264  1.247498  ...  5.597781  3.957320  5.775876
            2003         1.361571  3.489412  7.097156  ...  4.880698  3.832103  5.031273
            2004         2.264976  2.877810  4.056191  ...  3.253473  3.314512  4.896444
            2005         1.887321  3.466041  3.832519  ...  2.648234  4.212436  5.088535
            2006         1.456009  2.730801  2.910064  ...  3.306952  3.548342  5.672392
            2007         1.675358  2.575447  2.383468  ...  4.219317  3.666694  4.881267
            2008         1.786699  3.186896  1.782014  ...  2.834857  4.389405  6.141509
            2009         1.760088  3.462534  5.487852  ...  2.095825  3.013996  5.754901
        
            Where the columns represent the topics, each row a timestep and the cell values are the
            proportional relevance of a topic at a particular timestep.
            
        filename (str): Path which to save the plot figure to.
        title (str, optional): The title of the plot. Defaults to None.
        scale (int, optional): scale factor that dictates how fat the topic proportion representations are. 
            Defaults to 1.
        save (bool, optional): Whether or not to save the figure. Defaults to True.

    Returns:
        matplotlib plot object
    """
    # sns.set_context("talk")
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")
    sns.set_theme(
        palette=palette, 
        style={'axes.spines.bottom': True,
                'axes.grid':True,
                'axes.spines.left': False,
                'axes.spines.right': False,
                'axes.spines.top': False,
                'ytick.left': False,
                'figure.facecolor':'w'}, 
        context="paper"
    )
    fig = plt.figure(figsize=(30, 2 * len(df.columns)))
    ax = fig.gca()

    plt.yticks([])
    x_domain = [x for x in range(1,len(df)+1)]
    x_labels = df.index.tolist()
    assert len(x_domain) == len(x_labels)
    plt.xticks(x_domain, fontsize=fontsize)
    ax.set_xticklabels(x_labels)
    max_val = scale * df.max().max() + 5

    for i, t in enumerate(reversed(df.columns)):
        if use_colour_mapping:
            ind = colour_mapping.index(t)
            count = 0
            for col in cycle(sns.color_palette("bright", as_cmap=True)):
                if count == ind:
                    colour = col
                    break
                count += 1
            plt.fill_between(x_domain, df[t] + i*max_val, i*max_val - df[t], label=t, color=colour)
        else:
            plt.fill_between(x_domain, df[t] + i*max_val, i*max_val - df[t], label=t)
        plt.text(len(df) + 0.3, (i+0.) *max_val, t, fontsize=fontsize)

    plt.xlabel('Year', fontsize=fontsize)
    if title:
        plt.title(title)
    if save:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    return plt

def plot_word_ot(df, title, subplot=None, save_path=None):
    if subplot:
        plotter = subplot
    else:
        plotter = plt
    plotter.clf()
    fig = plotter.figure(figsize=(10,10))
    palette = sns.color_palette("viridis", as_cmap=True)
    sns.lineplot(data=df).set(title=title)
    if save_path:
        plotter.savefig(save_path)
    return plotter

def plot_word_ot_with_proportion(df, title, fig):
    """This function should plot words over time for a particular topic in a DTM with
    that topic's proportional relevance underneath it. We also want to be able to 
    plot multiple of these in subplots.

    Args:
        df ([type]): The dataframe containing the relevant over time data, similar to plot_word_ot, as well
        as a column outlining the proportional relevance of the topic over time. These two sets of data should have the same time range.
        title ([type]): [description]
        save_path ([type], optional): [description]. Defaults to None.
    """
    ax0, ax1 = fig.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [4,1]})
    ax1.set_xlabel('year', fontsize='xx-large')
    ax0.set_ylabel('word proportion', fontsize='xx-large')
    word_df = df.drop('_prop', axis='columns')
    prop_df = df['_prop']
    sns.lineplot(data=word_df, ax=ax0)
    ax0.legend(fontsize=20)
    single_topic_time_evolution(prop_df, ax1)

def plot_word_topic_evolution_ot(dfs, titles, figsize=None, save_path=None):
    if not figsize:
        fig = plt.figure(figsize=(len(dfs)*4,4))
    else:
        fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1,len(dfs))
    plt.subplots_adjust(bottom=0.06, top=1, left=0.08, right=1)
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")
    sns.set_theme(
        palette=palette, 
        style={'axes.spines.bottom': True,
                'axes.grid':True,
                'axes.spines.left': False,
                'axes.spines.right': False,
                'axes.spines.top': False,
                'ytick.left': False,
                'figure.facecolor':'w'}, 
        context="paper"
    )
    if len(dfs) > 1:
        for i,df in enumerate(dfs):
            plot_word_ot_with_proportion(df, titles[i], subfigs[i])
    else:
        plot_word_ot_with_proportion(dfs[0], titles[0], subfigs)
    fig.subplots_adjust(hspace=0.)
    if save_path:
        plt.savefig(save_path)

