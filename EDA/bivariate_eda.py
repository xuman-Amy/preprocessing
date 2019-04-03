import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

import warnings; warnings.filterwarnings(action='ignore')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (5,3),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
# %matplotlib inline

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


class BivariateEDA(object):
    """
    对于dataframe数据中两个字段的数据可视化。
    对于两个类别型（category）数据，可以使用堆叠直方图（plot_stackHist）,，;
    对于数值型（numeric）数据，可以使用联合分布图（plot_joint）
    """
    
    def __init__(self, data):
        """
        数据初始化，必须传入dataframe文件或者文件的有效路径。
        """
        #数值型类型
        self._numeric_type = ['int', 'int32', 'int64', 'float', 'float32', 'float64']
        
        #归一化函数
        self._log_scaler = lambda x: np.log10(x) / np.log10(max(x))
        self._min_max_scaler = lambda x:(x-np.min(x))/(np.max(x) - np.min(x))
        self._z_score_scaler = lambda x: (x - np.mean(x)) / np.std(x)
        self._no_scale = lambda x: x
        
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data,str) and os.path.exists(data):
            self.fpath = data
            self.data = self._read_data()
        else:
            print('请输入dataframe文件')
            return 
        return
    
    def _read_data(self):
        """
        如果传入的是文件路径，自动读取文件。
        """
        self.data = pd.read_csv(self.fpath)
        return self.data
    
    def plot_stackHist(self, x_category, y_category, figsize=params['figure.figsize'], ylim=None):
        """
        绘制两个类别型字段的堆叠直方图。
        
        x_category: x坐标轴上的类别
        y_category: 在x_category每个类别的分布
        figsize：（x,y）代表图显示的x和y坐标。
        ylim: y轴的坐标限制。

        """

        plt.figure(figsize=figsize, dpi= 80)
        df_agg = self.data.loc[:, [x_category, y_category]].groupby(y_category)
        vals = [df[x_category].values.tolist() for _, df in df_agg]
        colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
        n, bins, patches = plt.hist(vals, 
                            self.data[x_category].unique().__len__(), 
                            stacked=True, 
                            density=False, 
                                    color=colors[:len(vals)])

        plt.legend({group:col for group, col in 
                    zip((self.data[y_category]).unique().tolist(), colors[:len(vals)])},
                  fontsize=10)
        plt.gca().set(title="Stacked Histogram of %s by %s" % (x_category, y_category),
                      xlabel=x_category,
                      ylabel=y_category + '    Frequency',
                      ylim=ylim)
        plt.xticks(ticks=bins, 
                   labels=(self.data[x_category]).unique().tolist(), 
                   rotation=90, 
                   horizontalalignment='left')
        plt.show()
        
    def plot_scatter(self, x_category, y_category, group_category=None, 
                    minMaxScale=False, zScoreScale=False, logScale=False,
                     figsize = params['figure.figsize'], xlim = None, ylim = None, toIgnoreCate=None):
        """
        绘制两个字段间的散点图
        
        x_category: x坐标轴上的类别
        y_category: y坐标轴上的类别
        group_category: 显示在x,y上分布的类别
        figsize：（x,y）代表图显示的x和y坐标。
        xlim: x轴的坐标限制
        ylim: y轴的坐标限制
        toIgnoreCate：不显示的类别
        minMaxScale:对数据进行min-max归一化
        zScoreScale: 对数据进行z-score归一化
        logScale:对数据进行log归一化
        """
        if minMaxScale: 
            scaler = self._min_max_scaler
            scaler.name = 'minMaxScale'
        elif zScoreScale: 
            scaler = self._z_score_scaler
            scaler.name = 'zScoreScale'
        elif logScale:
            scaler = self._log_scaler
            scaler.name = 'logScale'
        else:
            scaler = self._no_scale
            scaler.name = 'noScale'
        if pd.isnull(self.data[x_category]).sum() > 0:
            self.data = self.data[pd.notnull(self.data[x_category])]
        
        if group_category != None:
            categories = np.unique(self.data[group_category])
            colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
            plt.figure(figsize=figsize, dpi= 80, facecolor='w', edgecolor='k')
            for i, c in enumerate(categories):
                if c != toIgnoreCate:
                    data = self.data.loc[self.data[group_category] == c,:]
                    plt.scatter(data[[x_category]].apply(scaler),data[[y_category]].apply(scaler),
                            s=20, color=colors[i], label=c)
        else:
            plt.scatter(self.data[x_category], self.data[y_category])
        plt.gca().set(xlabel=x_category, ylabel=y_category,
                      xlim=xlim, ylim=ylim)
        plt.xticks(fontsize=12, rotation=90);
        plt.yticks(fontsize=12)
        plt.title("Scatterplot of" +  x_category +  "vs" + y_category, fontsize=22)
        plt.legend(fontsize=12,loc = 'upper right')    
        plt.show() 
        
    def plot_box(self, x_category, y_category, figsize = params['figure.figsize'], 
                 minMaxScale=False, zScoreScale=False, logScale=False,
                 ylim = None, add_n_obs_state = False):
        """
        绘制两个类别型字段间的箱型图
        x_category：
        y_category：
        figsize：（x,y）代表图显示的x和y坐标。
        ylim: y轴的坐标限制
        add_n_obs_state：是否显示箱型图中分布的点
        minMaxScale:对数据进行min-max归一化
        zScoreScale: 对数据进行z-score归一化
        logScale:对数据进行log归一化
        """
        plt.figure(figsize = figsize, dpi= 80)
        
        # 数据归一化
        if minMaxScale: 
            scaler = self._min_max_scaler
            scaler.name = 'minMaxScale'
        elif zScoreScale: 
            scaler = self._z_score_scaler
            scaler.name = 'zScoreScale'
        elif logScale:
            scaler = self._log_scaler
            scaler.name = 'logScale'
        else:
            scaler = self._no_scale
            scaler.name = 'noScale'
        if self.data[y_category].dtype in self._numeric_type:
            sns.boxplot(x=self.data[x_category], 
                        y=self.data[[y_category]].apply(scaler).iloc[:,0], 
                        notch=True)
            sns.stripplot(x=self.data[x_category], 
                          y=self.data[[y_category]].apply(scaler).iloc[:,0], 
                          color='black', size=3, jitter=1)

        def add_n_obs(df, group_col, y):
#             medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}
            xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
            n_obs = df.groupby(group_col)[y].size().values
            for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
                plt.text(x,0, 
                         "#obs : "+str(n_ob), 
                         rotation = 60,
                         horizontalalignment='left', 
                         fontdict={'size':10}, color='blue')
        if add_n_obs_state :
            add_n_obs(self.data, group_col=x_category, y=y_category)
            
        plt.xticks(rotation=90)
        plt.gca().set(title=y_category + '    Box', ylim=ylim)
        plt.show()
        
    def plot_kmeans_cluster(self, x_category, y_category, n_clusters=3, figsize=params['figure.figsize'], 
                            ylim=None, xlim=None):
        """
        绘制k-means聚类后的数据分布
        
        x_category：
        y_category：
        n_clusters:聚类的簇数量
        figsize：（x,y）代表图显示的x和y坐标。
        xlim: x轴的坐标限制
        ylim: y轴的坐标限制
        """
        tmp = np.array(self.data[[x_category,y_category]])
        kms = KMeans(n_clusters=n_clusters)
        clf = kms.fit_predict(tmp)
        plt.figure(figsize=figsize)
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        for i in range(0,len(clf)):
            plt.plot(tmp[i,0],tmp[i,1], mark[clf[i]])
            
        plt.gca().set(xlabel=x_category, ylabel=y_category, 
                      xlim=xlim, ylim=ylim)
        plt.show()
        
    def plot_counts(self, x_category, y_category, delColumn=None, 
                   bigSize=2, figsize=params['figure.figsize'], xlim=None, ylim=None):
        """
        绘制两个类别型数据间的数量显示
        
        x_category：类别型字段
        y_category：类别型字段
        delColumn: x_category中不显示的字段
        bigSize:放大倍数（数量乘以bigsize是最后的大小）
        figsize：（x,y）代表图显示的x和y坐标。
        xlim: x轴的坐标限制
        ylim: y轴的坐标限制
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=80)
        data_counts = self.data[self.data[x_category]!=delColumn ].groupby([x_category, y_category])\
                            .size().reset_index(name='counts')
        sns.stripplot(x=data_counts[x_category],
                      y=data_counts[y_category], 
                      size=data_counts['counts'] * bigSize, 
                      ax=ax)
        plt.legend(loc='left')
        plt.xticks(rotation = 90)
        plt.gca().set(title = 'Counts Plot ', xlim=xlim, ylim=ylim)
        plt.show()
        
    def plot_categoty_inside(self, x_category, y_category, col_wrap=6,
                             kind='count',figsize=(10,6), dpi=80):
        """
        绘制某一字段内各个类别在另一个类别上的数据统计显示
        
        x_category：类别型字段
        y_category：类别型字段
        col_wrap：每一行显示的小图个数
        kind：显示的类型，‘counts’,'box','violin'等
        figsize：（x,y）代表图显示的x和y坐标。
        """
        
        plt.figure(figsize=figsize, dpi=dpi)
        g = sns.catplot(x=x_category, col=y_category, 
                        col_wrap=col_wrap, data=self.data,
                        kind=kind, height=4, aspect=1, 
                        palette='tab20')
#         plt.title(y_category + ' VS ' + x_category, loc='center')
        plt.show()
    
    def plot_ordered_bar(self, x_category, y_category, mean=False, figsize=params['figure.figsize']):
        """
        绘制数值型字段在类别型字段的各个类别上的平均值
        
        x_category：类别型字段
        y_category：数值型字段
        figsize：（x,y）代表图显示的x和y坐标。
        """
        if mean:
            mean = lambda x : x.mean()
        else:
            mean = lambda x : x
        df = self.data[[x_category, y_category]].groupby(x_category).apply(mean)
        df.sort_values(y_category, inplace=True)
        if self.data[x_category].dtype != 'object':
                df.reset_index(drop=True, inplace=True)
        df.reset_index(inplace=True)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi= 80)
        ax.vlines(x=df.index, ymin=0, ymax=df[y_category], color='firebrick', alpha=0.7, linewidth=20)

        for i, sim in enumerate(df[y_category]):
            ax.text(i, sim+0.2, round(sim, 1), 
                    horizontalalignment='center',
                    fontdict = {'fontsize':14})

        ax.set_title('Bar Chart for '+y_category, fontdict={'size':22})
        ax.set(ylabel=y_category)
        plt.xticks(df.index, df[x_category], rotation=60, 
                   horizontalalignment='right', fontsize=12)

        p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, 
                               alpha=.1, facecolor='green', transform=fig.transFigure)
        p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, 
                               alpha=.1, facecolor='red', transform=fig.transFigure)
        fig.add_artist(p1)
        fig.add_artist(p2)
        plt.show()
        
    def plot_joint(self, x_category, y_category, kind='reg', color='#C44E52',ratio=3,
                   minMaxScale=False, zScoreScale=False, logScale=False,
                   figsize=params['figure.figsize'], xlim=None, ylim=None):
        """
        绘制两个数值型数据的联合分布
        
        x_category:数值型字段
        y_category: 数值型字段
        kind：‘scatter’‘hex’‘reg’‘kde’
        retio：中心图与侧边的比例
        minMaxScale:对数据进行min-max归一化
        zScoreScale: 对数据进行z-score归一化
        logScale:对数据进行log归一化
        """
        
        plt.figure(figsize=figsize, dpi=80)
         # 数据归一化
        if minMaxScale: 
            scaler = self._min_max_scaler
            scaler.name = 'minMaxScale'
        elif zScoreScale: 
            scaler = self._z_score_scaler
            scaler.name = 'zScoreScale'
        elif logScale:
            scaler = self._log_scaler
            scaler.name = 'logScale'
        else:
            scaler = self._no_scale
            scaler.name = 'noScale'
        if self.data[y_category].dtype in self._numeric_type:
            g = sns.jointplot(self.data[[x_category]].apply(scaler), self.data[[y_category]].apply(scaler),
                              color=color, ratio=3, kind=kind)
        plt.gca().set(xlim=xlim, ylim=ylim,
                     xlabel=x_category, ylabel=y_category)
    
#         plt.title('jointplot of '+ x_category + 'VS'+ y_category, loc='right')
        plt.show()
    def plot_distribution( self, x_category, y_category , size=2, **kwargs, ):
        """
        分布图
        
        x_category: x轴
        y_category: y轴
        row： 横向对比
        col：纵向对比
        """
        row = kwargs.get( 'row' , None )
        col = kwargs.get( 'col' , None )
        facet = sns.FacetGrid(data=self.data, 
                              hue=y_category, 
                              aspect=4, row=row, 
                              col=col,
                              size=size
                             )
        facet.map(sns.kdeplot, x_category, shade= True )
        facet.set(xlim=(0, self.data[x_category].max()))
        facet.add_legend()
if __name__ == '__main__':
    fpath = './data/data2.csv'
    plotUni = BivariateEDA(fpath)
    x = "B_POI_Category"
    y = "B_POI_Hitcount"
    plotBivar.plot_box(x_category = x ,y_category = y, logScale=True, add_n_obs_state=True)
